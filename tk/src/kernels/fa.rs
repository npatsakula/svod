//! Flash-attention forward — a hand-authored forward kernel using online softmax
//! and a double-buffered K/V stream (forward only; no backward pass).
//!
//! One workgroup (single wave64 warp) owns one `(head, q_block, batch)` triple:
//! it loads its Q tile into registers, then streams the K/V blocks, computing
//! `QKᵀ` with [`mma_atb`](crate::Group::mma_atb), applying the causal mask, the
//! running-max online softmax (the LDS cross-lane [`row_reduce`]s), and the `A·V`
//! accumulation, before normalizing and writing the transposed output tile back.
//!
//! The K/V stream is arch-selected inside [`build_fa_mw_rdb`]: register-staged
//! (`global_load` early, `ds_write` late) on AMD, `cp.async` into the other LDS half
//! with the copy in flight under the current block's compute on CUDA sm_80+, where
//! the LDS→register gathers are `ldmatrix.x4` (see [`crate::Group::load`]).

use std::sync::Arc;

use smallvec::smallvec;
use snafu::ensure;
use svod_dtype::DType;
use svod_ir::{ConstValue, UOp};
use svod_tensor::Tensor;

use crate::Group;
use crate::group::MoveIdx;
use crate::index::{Idx, load_at};
use crate::kernel::Kernel;
use crate::loop_scope::Loop;
use crate::scaffold::GlSpec;
use crate::tile::{RT, RV, RegTile, ST};
use crate::tiles::TileLayout;
use svod_codegen::llvm::nvptx::smem::{cp_async_wait, cp_async_wait_all};

/// The WMMA tile edge (gfx942 K=16). The QKᵀ / A·V WMMAs always operate on
/// 16×16 fragments; Q/KV per-warp *tiles* are grids of `BLK`-edged fragments
/// ([`Q_BLK`]/[`KV_BLK`]).
const BLK: usize = 16;

/// Multi-wave warps per workgroup (the multi-wave occupancy lift, 8 waves/block):
/// `8 * wave_size` threads per block (512 at wave64, 256 at warp32). Each warp owns
/// a distinct Q-tile; all 8 share one K/V LDS slot, filled collaboratively across
/// the block.
pub(crate) const NUM_WARPS: usize = 8;

/// Default per-warp Q-tile height for the production double-buffered path
/// ([`flash_attention_forward_mw_db`]). The default `{16,16}` Q/KV tile (the WMMA
/// edge) is tuned for gfx942 register occupancy; larger tiles raise VGPR pressure
/// and drop occupancy (the bottleneck). The multi-wave occupancy lift (8
/// waves/block) is opt-in via [`FaConfig`]. `{32,32}`/`{32,64}` stay
/// opt-in via the explicit-tile [`build_fa_mw_db`] args.
const Q_BLK: usize = 16;
/// Sequence-length multiple accepted by the baseline production flash-attention
/// tile. Callers that explicitly choose to pad can use this without duplicating
/// the kernel's tiling details.
pub const FLASH_ATTENTION_SEQUENCE_MULTIPLE: usize = Q_BLK * NUM_WARPS;
/// Default per-warp KV super-block height. `32` (2·BLK): profiling the small-grid
/// fallback (the b=1/h=16 inference regime) showed FA is ILP/recurrence-bound, not
/// occupancy-bound — a taller KV super-block raises per-warp WMMA ILP and halves the
/// KV passes (fewer online-softmax bookkeeping ops), winning ~5% at n=1024 and ~12%
/// at n=2048 even as occupancy drops 50→38%. (`q_blk` stays `16`: a taller Q-tile
/// instead halves the launch grid → fewer waves → slower.)
const KV_BLK: usize = 32;

fn iconst(v: i64) -> Arc<UOp> {
    UOp::index_const(v)
}

/// The GPU arch(es) the **production graph** flash-attention ([`flash_attention_with`]
/// → [`build_fa_mw_rdb`]) is enabled for: gfx942 (CDNA MFMA, wave64), gfx1151
/// (RDNA3.5 WMMA, wave32) and CUDA sm_80+ (`mma.sync`, warp32). The launcher gates
/// against this list; generic launch infrastructure stays architecture-agnostic.
pub const FA_SUPPORTED_ARCHS: crate::ArchSet =
    crate::ArchSet::amd(&[svod_dtype::AmdArch::Gfx942, svod_dtype::AmdArch::Gfx1151])
        .with_cuda_from(svod_dtype::CudaArch::from_compute_capability(8, 0));

/// Whether `device` can run the production graph flash-attention kernel.
/// Uses the same architecture and toolchain gate as [`crate::launch_custom`].
pub fn flash_attention_supported(device: &svod_dtype::DeviceSpec) -> bool {
    crate::target::resolve_supported_arch(device, FA_SUPPORTED_ARCHS).is_ok()
}

/// The **direct-launch** FA wrappers ([`flash_attention_forward`], `_mw`, `_mw_db`,
/// `_mw_rdb`) hardcode the wave64 block size and the CDNA fragment tiles, so they
/// stay gfx942-only.
const FA_DIRECT_SUPPORTED_ARCHS: crate::ArchSet = crate::ArchSet::amd(&[svod_dtype::AmdArch::Gfx942]);

/// Validate a direct-launch wrapper's device against [`FA_DIRECT_SUPPORTED_ARCHS`].
fn fa_check_target(t: &Tensor) -> crate::LaunchResult<()> {
    crate::target::check_target(&t.device(), FA_DIRECT_SUPPORTED_ARCHS)
}

/// Tuning knobs for [`build_fa_mw_rdb`] — the structured replacement for its former
/// positional `bool`/tile args (mirrors [`crate::kernels::matmul::MatmulCfg`]).
/// [`Default`] is the production baseline: `{16,16}` per-warp tile, rolled (looped)
/// causal compute. The shape (`b,n,h,h_kv,d`) stays a positional arg since it's
/// derived from the input tensors, not a tuning choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaConfig {
    /// Per-warp Q-tile height (a multiple of the WMMA edge `16`). A value not a
    /// multiple of 16 panics the builder (the divisibility assert).
    pub q_blk: usize,
    /// Per-warp KV super-block height (a multiple of `16`). A value not a multiple
    /// of 16 panics the builder (the divisibility assert).
    pub kv_blk: usize,
    /// Emit the fully-unrolled (flat) QKᵀ/softmax/A·V body instead of the rolled loop.
    pub unroll: bool,
    /// Causal masking + KV block-skip. `false` is the full (bidirectional) attention
    /// sweep over every KV super-block.
    pub causal: bool,
}

impl Default for FaConfig {
    fn default() -> Self {
        Self { q_blk: Q_BLK, kv_blk: KV_BLK, unroll: false, causal: true }
    }
}

/// Online-softmax state carried *across* KV iterations (the [`build_fa_mw_rdb`]
/// loop's back-edge re-reads the rewrapped handles each trip).
struct FaAcc<'k> {
    max_vec: RV<'k>,
    norm_vec: RV<'k>,
    o_reg: RT<'k>,
}

/// Per-warp scratch register tiles for one KV super-block of [`build_fa_mw_rdb`]:
/// the K/V fragments, the `QKᵀ` accumulator + its WMMA-input cast, and the
/// online-softmax `max_vec_last`.
struct FaScratch<'k> {
    k_reg: RT<'k>,
    k_reg_t: RT<'k>,
    v_reg: RT<'k>,
    att: RT<'k>,
    att_mma: RT<'k>,
    max_vec_last: RV<'k>,
    /// RDNA-only per-warp LDS scratch (`[NUM_WARPS·kv_blk, q_blk]`) for the
    /// `att → att_mma` accumulator→input relayout. `None` on gfx942, where the
    /// accumulator and WMMA-input fragment layouts coincide so a register `copy`
    /// suffices; `Some` on gfx11, where they differ and the relayout must round-trip
    /// through LDS (store the even/odd accumulator, reload as the replicated input).
    att_smem: Option<ST>,
}

/// Loop-invariant context for the [`build_fa_mw_rdb`] KV-slice helpers: the per-warp
/// geometry + masking config + the warp [`Group`]/[`Loop`] handles every slice
/// shares. Bundling these keeps [`fa_qk`]/[`fa_softmax_pv`] to their per-slice
/// tiles and indices instead of long positional arg lists.
struct FaCtx<'a, 'k> {
    warp: &'a Group<'k>,
    lp: &'a Loop<'k>,
    q_reg_t: &'a RT<'k>,
    q_blk: &'a Arc<UOp>,
    warpid: &'a Arc<UOp>,
    causal: bool,
    valid_len: Option<Arc<UOp>>,
}

/// Apply the FA score-mask (causal + optional padding) to the `att` tile. The
/// causal mask zeros (via `−∞`) keys ahead of this warp's own query rows
/// (`kv_pos > q_pos`); the padding mask zeros keys at/after the per-batch
/// valid length (`kv_pos >= valid_len`). With neither, the tile is returned
/// unchanged (the early-return avoids emitting any mask IR when masking is off).
/// The per-element `(kv_pos, q_pos)` is computed arch-correctly inside
/// [`Group::mask_where`].
fn score_mask<'k>(
    warp: &Group<'k>,
    att: RT<'k>,
    slice_idx: &Arc<UOp>,
    q_blk: &Arc<UOp>,
    causal: bool,
    valid_len: Option<&Arc<UOp>>,
) -> RT<'k> {
    if !causal && valid_len.is_none() {
        return att;
    }
    let row_blk = Idx::Uop(slice_idx.clone());
    let col_blk = Idx::Uop(q_blk.clone());
    let att = if causal {
        warp.mask_where(att, row_blk.clone(), col_blk.clone(), f64::NEG_INFINITY, |kv_pos, q_pos| kv_pos.gt(q_pos))
    } else {
        att
    };
    if let Some(vl) = valid_len {
        warp.mask_where(att, row_blk, col_blk, f64::NEG_INFINITY, move |kv_pos, _| kv_pos.ge(vl))
    } else {
        att
    }
}

/// Stage 1 of a KV slice — `QKᵀ`: gather this warp's K/V fragments from the
/// already-filled shared `(k_smem, v_smem)` LDS, compute `QKᵀ` into a
/// freshly-zeroed `att`, and apply the causal mask. Returns the masked raw scores
/// `att` and the gathered `v_reg` (carried to [`fa_softmax_pv`]). Splitting QK off
/// the softmax/PV lets the cross-tile pipeline emit `qk(cur)` out of phase with
/// `softmax_pv(prev)`. `fence` (the register-staged stream) gates the LDS→REG read
/// behind a cross-wave WAR barrier with the double-buffer prefetch commits folded
/// in; the `cp.async` stream fences at the loop top instead and passes `None`.
#[allow(clippy::too_many_arguments)]
fn fa_qk<'k>(
    ctx: &FaCtx<'_, 'k>,
    k_reg: RT<'k>,
    k_reg_t: RT<'k>,
    v_reg: RT<'k>,
    att: RT<'k>,
    k_smem: ST,
    v_smem: ST,
    slice_idx: &Arc<UOp>,
    fence: Option<&[Arc<UOp>]>,
) -> (RT<'k>, RT<'k>) {
    let warp = ctx.warp;
    // Per-warp LDS→REG gather: every warp reads the shared K/V block.
    let k_reg = warp.load(k_reg, k_smem, MoveIdx::default());
    let v_reg = warp.load(v_reg, v_smem, MoveIdx::default());
    // Cross-wave WAR sync: all 8 warps must finish reading this buffer before the
    // next fill overwrites it. The extra deps fold in the rolled double-buffer's
    // prefetch commits, so this single in-loop barrier (consumed by the gathers)
    // also gates the cross-iteration RAW/WAR.
    let (k_reg, v_reg) = match fence {
        Some(extra) => warp.war_fence2(k_reg, v_reg, extra),
        None => (k_reg, v_reg),
    };

    // QKᵀ into a freshly-zeroed att tile (re-zeroed each trip via the loop scope).
    let att = warp.zero(ctx.lp.reinit(att));
    let k_reg_t = warp.transpose(k_reg_t, &k_reg);
    let att = warp.mma_atb(att, &k_reg_t, ctx.q_reg_t);

    let att = score_mask(warp, att, slice_idx, ctx.q_blk, ctx.causal, ctx.valid_len.as_ref());
    (att, v_reg)
}

/// Stage 2 of a KV slice — online softmax + `A·V`: given the masked raw scores
/// `att` (from [`fa_qk`]) and the gathered `v_reg`, update the running max,
/// rescale the running stats by `exp2(prev_max - new_max)`, exponentiate, fold the
/// norm, and accumulate `A·V` into `o_reg`. Threads the updated [`FaAcc`] out.
///
/// `att` is col-layout `(KV=height, Q=width)`; softmax reduces over KV and
/// broadcasts per Q, so the reduce folds the *height* (KV) via [`Group::col_reduce`]
/// → a per-*width* (Q) vector. At `{16,16}` this is bit-identical to `row_reduce`;
/// for multi-fragment tiles it is the only orientation that folds the right axis.
fn fa_softmax_pv<'k>(
    ctx: &FaCtx<'_, 'k>,
    acc: FaAcc<'k>,
    att_mma: RT<'k>,
    att_smem: Option<ST>,
    max_vec_last: RV<'k>,
    att: RT<'k>,
    v_reg: &RT<'k>,
) -> FaAcc<'k> {
    let (warp, lp) = (ctx.warp, ctx.lp);
    let FaAcc { mut max_vec, mut norm_vec, mut o_reg } = acc;

    let max_vec_last = warp.copy(lp.reinit(max_vec_last), &max_vec);
    max_vec = warp.col_reduce(max_vec.after(&max_vec_last), &att, |a, b| a.max(b), f64::NEG_INFINITY);

    // Online-softmax rescale `exp2(prev_max - new_max)` as a same-shape vec−vec op
    // — reuses `max_vec_last`'s buffer (dead after this), so no scratch `scale_vec`
    // and no hand-rolled `load_at` merge.
    let scale_vec = (max_vec_last - &max_vec).exp2();

    o_reg = o_reg * &scale_vec;
    norm_vec = norm_vec * &scale_vec;

    let att = (att - &max_vec).exp2();

    norm_vec = warp.col_reduce(norm_vec.after(&scale_vec), &att, |a, b| a.add(b), 0.0);

    // Relayout the softmax weights `att` (the QKᵀ f32 accumulator) into the WMMA
    // input `att_mma` for the A·V matmul. On gfx942 the MFMA accumulator and input
    // fragments share a layout, so a register `copy` (with the f32→in-dtype cast)
    // suffices. On gfx11 they differ (even/odd `<8×f32>` acc vs replicated `<16×in>`
    // input), so a register copy is wrong — round-trip through this warp's LDS band:
    // store the even/odd accumulator (matrix `(kv,q)` order), barrier, reload with
    // the replicated-input map (`K=kv=element`, `N=q=lane%16`). Both lane maps are
    // the matmul-validated ones, so the relayout is correct by construction.
    let att_mma = match att_smem {
        None => warp.copy(att_mma.after((lp.index(), &norm_vec)), &att),
        Some(att_smem) => {
            // This warp's `(kv_blk × q_blk)` band of the shared relayout buffer, as a
            // zero-copy subtile — so the store and the reload address the warp's band
            // with no repeated wave-block index (mirrors the matmul LDS gather). The
            // band size is `att`'s element shape (its fragment grid × the base edge).
            let an = att.shape().len();
            let dims = (att.shape()[an - 3] * att.base.base.rows, att.shape()[an - 2] * att.base.base.cols);
            let band = att_smem.subtile(dims, (ctx.warpid.clone(), 0));
            let stored = warp.store(band, att, MoveIdx::default());
            let bar = stored.uop().barrier(smallvec![lp.index().clone(), norm_vec.uop().clone()]);
            let stored = stored.rewrap(stored.uop().after(smallvec![bar]));
            warp.load(att_mma, stored, MoveIdx::default())
        }
    };
    o_reg = warp.mma_atb(o_reg, v_reg, &att_mma);

    FaAcc { max_vec, norm_vec, o_reg }
}

// =============================================================================
// Software-pipelined double-buffered KV loop.
// =============================================================================

/// Software-pipelined double-buffered multi-wave flash-attention. Same
/// grid/semantics as [`build_fa_mw_db`], but the KV loop is a rolled `Range` over a
/// **2×-size LDS** K/V double buffer indexed by `kv_idx % 2`: each iteration
/// register-stages the next KV block's GLOBAL→VGPR load, gathers the current buffer
/// half into the WMMA fragments, runs the online-softmax body, then `ds_write`-
/// commits the staged registers into the other half, under one workgroup barrier
/// per iteration.
///
/// Unlike the unroll-by-2 [`build_fa_mw_db`] (two static buffers, two slices and
/// two [`FaScratch`] sets per body), this keeps one scratch set and one loop body.
/// LDS is the same (one `st_db` = two halves); FA's K/V are small so 2× fits the
/// 64 KB budget. The online-softmax [`FaAcc`] carries across the back-edge via the
/// memory-accumulator (`kv_idx` re-init) pattern, as in [`build_fa_mw`]. The
/// `kv_idx % 2` parity makes the gather/commit counter-dependent so they stay
/// loop-scoped; the per-iteration WAR barrier (consumed by the gathers, with the
/// prefetch commits folded into its deps) provides the cross-iteration RAW/WAR
/// ordering, closed with plain [`Kernel::endrange`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_fa_mw_rdb(
    ker: &Kernel,
    b: usize,
    n: usize,
    h: usize,
    h_kv: usize,
    d: usize,
    cfg: FaConfig,
    in_dtype: DType,
    masked: bool,
) {
    let FaConfig { q_blk: q_blk_rows, kv_blk: kv_blk_rows, unroll, causal, .. } = cfg;
    // Flat compute (unrolled QKᵀ/softmax/A·V) is the prerequisite for the Stage-2
    // attention scheduling comb; the rolled (`unroll = false`) form is the iglp
    // baseline. Same numerics either way (the unroll only changes the loop
    // mechanism, not the fold order).
    ker.set_unroll(unroll);
    Kernel::assert_divisible(d, BLK, "FA D");
    Kernel::assert_divisible(q_blk_rows, BLK, "FA Q_BLK");
    Kernel::assert_divisible(kv_blk_rows, BLK, "FA KV_BLK");
    Kernel::assert_divisible(h, h_kv, "FA H / H_KV");
    Kernel::assert_divisible(n, q_blk_rows * NUM_WARPS, "multi-wave FA N");
    // Rolled (no unroll halving): the group-max causal bound is
    // `(block_q_base+1)*NUM_WARPS*Q_BLK/KV_BLK` super-blocks (exact for these tiles).
    Kernel::assert_divisible(NUM_WARPS * q_blk_rows, kv_blk_rows, "FA rolled-db KV_BLK");
    let group_size = (h / h_kv) as i64;
    let g = ker.group(NUM_WARPS);
    let warp = ker.warp();

    // ABI: outputs (o) then inputs (q, k, v), fixed by construction.
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[b, n, h, d], in_dtype.clone())],
        &[
            GlSpec::new(&[b, n, h, d], in_dtype.clone()),
            GlSpec::new(&[b, n, h_kv, d], in_dtype.clone()),
            GlSpec::new(&[b, n, h_kv, d], in_dtype.clone()),
        ],
    );
    let (o, q, k, v) = (outs[0].clone(), ins[0].clone(), ins[1].clone(), ins[2].clone());
    // Per-batch valid key-length buffer (padding mask), bound AFTER o,q,k,v (trailing —
    // never interleaved) so the ABI slot order stays stable; only bound when `masked`.
    // The scalar `lens[batch]` is already int32, matching the concrete SPECIAL
    // position arithmetic.
    let valid_len = masked.then(|| {
        let lens = ker.gl(&[b], DType::Int32);
        load_at(lens.uop(), lens.shape(), &[Idx::from(&ker.block_idx[2])])
    });

    let head = ker.grid_x();
    let head_kv = head.floor_div(&iconst(group_size));
    let batch = ker.grid_z();
    let block_q_base = ker.grid_y();
    let warpid = g.warpid_in_group();
    let q_blk = block_q_base.mul(&iconst(NUM_WARPS as i64)).add(&warpid);

    let in_dt = in_dtype.clone();
    let f32 = DType::Float32;
    let (row, col) = (TileLayout::Row, TileLayout::Col);

    // Tiles below are declared by ROLE via the scaffold shortcuts (`ker.acc`/`operand`/
    // `acc_t`/`shared_db`/`shared`), which resolve the arch fragment through `caps.frag`
    // — so the kernel never names a physical fragment constant. `att_smem` (below) is the
    // per-warp LDS relayout band, needed only where the accumulator cannot be reused as a
    // WMMA input (RDNA); on CDNA the fragments coincide and the relayout is a register copy.

    // 2×-size shared K/V LDS double buffers (one `kv_blk_rows × d` block per half).
    let k_smem = ker.shared_db((kv_blk_rows, d), in_dt.clone(), row);
    let v_smem = ker.shared_db((kv_blk_rows, d), in_dt.clone(), row);
    let half_k = k_smem.half_elems() as i64;
    let half_v = v_smem.half_elems() as i64;

    // Q tile + transpose (shared, read-only across the loop). `o_reg_t` is the
    // transpose of the `[d,q]` PV accumulator for the `O[q,d]` store (N-major ⇒
    // `rt_acc_t` on RDNA).
    let q_reg_fl = ker.operand((q_blk_rows, d), f32, row);
    let q_reg = ker.operand((q_blk_rows, d), in_dt.clone(), row);
    let q_reg_t = ker.operand((d, q_blk_rows), in_dt.clone(), col);
    let o_reg_t = ker.acc_t((q_blk_rows, d), row);

    // One scratch set: the rolled body has a back-edge, so the carried FaAcc + a
    // single set suffice. `att_smem` holds one per-warp relayout band on RDNA.
    let sc = FaScratch {
        k_reg: ker.operand((kv_blk_rows, d), in_dt.clone(), row),
        k_reg_t: ker.operand((d, kv_blk_rows), in_dt.clone(), col),
        v_reg: ker.operand((kv_blk_rows, d), in_dt.clone(), col),
        att: ker.acc((kv_blk_rows, q_blk_rows), col),
        att_mma: ker.operand((kv_blk_rows, q_blk_rows), in_dt.clone(), col),
        max_vec_last: ker.acc_vec(q_blk_rows),
        att_smem: (!ker.caps.acc_reusable_as_input())
            .then(|| ker.shared((NUM_WARPS * kv_blk_rows, q_blk_rows), in_dt.clone(), row)),
    };

    // Carried online-softmax accumulators.
    let o_reg = ker.acc((d, q_blk_rows), col);
    let max_vec = ker.acc_vec(q_blk_rows);
    let norm_vec = ker.acc_vec(q_blk_rows);
    let acc = FaAcc { max_vec: warp.neg_inf_rv(max_vec), norm_vec: warp.zero_rv(norm_vec), o_reg: warp.zero(o_reg) };

    // Load + scale this warp's Q tile, then transpose for the QKᵀ contraction.
    let q_reg_fl = warp.load(q_reg_fl, q, MoveIdx::block((batch.clone(), q_blk.clone(), head.clone(), 0), 1));
    let q_reg_fl = q_reg_fl * ((1.0 / (d as f64).sqrt()) * std::f64::consts::LOG2_E);
    let q_reg = warp.copy(q_reg, &q_reg_fl);
    let q_reg_t = warp.transpose(q_reg_t, &q_reg);

    // Total KV super-blocks (the full bidirectional sweep). With `causal`, the
    // per-q-block bound is the causal block-skip `(block_q_base+1)*NUM_WARPS*Q_BLK/KV_BLK`
    // super-blocks; without it every q-block attends to all `total_kv_blocks`.
    let total_kv_blocks = (n / kv_blk_rows) as i64;
    let kv_bound = if causal {
        let blocks_mult = (NUM_WARPS * q_blk_rows / kv_blk_rows) as i64;
        block_q_base.add(&iconst(1)).mul(&iconst(blocks_mult))
    } else {
        iconst(total_kv_blocks)
    };

    // The K/V stream: `cp.async` (sm_80+ — the copy lands in LDS with no register
    // staging and stays in flight under the previous block's compute) or the
    // register-staged prefetch (AMD).
    let async_stream = g.cp_async_fill_applies(&k_smem, &k) && g.cp_async_fill_applies(&v_smem, &v);

    // Prologue: block 0 → buf[0]. Register-staged: stage → VGPR, commit, barrier;
    // cp.async: issue + commit only (the loop top retires and fences it).
    let p_kidx = [Idx::from(&batch), Idx::Const(0), Idx::from(&head_kv), Idx::Const(0)];
    let (k_smem, v_smem) = if async_stream {
        let c_k = g.cp_async_fill(&k_smem, &k, &p_kidx, 1);
        let c_v = g.cp_async_fill(&v_smem, &v, &p_kidx, 1);
        (k_smem.after(c_k), v_smem.after(c_v))
    } else {
        let s0_k = g.stage_global_to_reg(&k_smem, &k, &p_kidx, 1);
        let s0_v = g.stage_global_to_reg(&v_smem, &v, &p_kidx, 1);
        (g.commit_reg_to_local(k_smem, &s0_k, true), g.commit_reg_to_local(v_smem, &s0_v, true))
    };

    // Rolled KV loop. `kv_bound` (the dynamic per-q-block causal trip count) is the
    // Range end. The prefetch-block index is `(kv+1) % total_kv_blocks` (a FloorMod): the
    // final trip's prefetch (`kv+1 == total`) wraps to block 0, which is never
    // gathered, keeping the GLOBAL read in bounds. A `min`/`where` clamp is avoided
    // — a `WHERE` in the prefetch-address path is mis-ordered past its address-MUL
    // consumer in this kernel's linearization, leaving the renderer without its SSA
    // value; FloorMod (like the parity) lowers and orders cleanly.
    let lp = ker.loop_dynamic(kv_bound);
    let kv_idx = lp.index().clone();
    let kvp1 = kv_idx.add(&iconst(1));
    let pf = kvp1.try_mod(&iconst(total_kv_blocks)).expect("(kv+1) % total blocks");
    let par_cur = kv_idx.try_mod(&iconst(2)).expect("kv % 2");
    let par_nxt = kvp1.try_mod(&iconst(2)).expect("(kv+1) % 2");

    let k_cur = k_smem.with_base_offset(par_cur.mul(&iconst(half_k)));
    let v_cur = v_smem.with_base_offset(par_cur.mul(&iconst(half_v)));
    let k_nxt = k_smem.with_base_offset(par_nxt.mul(&iconst(half_k)));
    let v_nxt = v_smem.with_base_offset(par_nxt.mul(&iconst(half_v)));

    // Mark the KV-loop as an attention compute pipeline (MFMA + online softmax),
    // threaded through the in-loop K/V buffers so the marker precedes the first
    // prefetch load and stays loop-scoped (dep = `kv_idx`). The prologue keeps the
    // un-rewrapped `k`/`v`. The post-linearization scheduling pass brackets the MFMAs
    // and (Stage 2) weaves the softmax under them (supersedes the prior `iglp_opt(0)`).
    let pf_kidx = [Idx::from(&batch), Idx::from(&pf), Idx::from(&head_kv), Idx::Const(0)];
    let mark = crate::sched::pipeline(crate::sched::SchedKind::Attention, kv_idx.clone());
    let k_l = k.rewrap(k.uop().after(smallvec![mark.clone()]));
    let v_l = v.rewrap(v.uop().after(smallvec![mark]));

    // Per-iteration ordering of the two streams (one workgroup barrier each):
    //
    // cp.async — `wait_group 0` + barrier at the loop TOP retire block `kv` (issued
    // last iteration, or by the prologue) and prove every warp finished gathering
    // buf[nxt] last iteration (WAR); then block `kv+1` is issued into buf[nxt] and
    // stays in flight under this block's gather + compute (the gathers order after
    // the commit so the copy issues first). The final trip's wrapped prefetch is
    // drained after the loop.
    //
    // register-staged — stage block `kv+1` → VGPR, `ds_write` it into buf[nxt] (no
    // per-commit barrier; emitted before the slice so the slice's `o_reg` A·V store
    // stays the last terminal store on the stack), gather buf[cur], then the WAR
    // barrier consumed by the gathers folds in the commits, gating both the
    // cross-iteration RAW and WAR. The barrier-wrapped END (`endrange_barrier_to`)
    // is NOT used: it reorders the causal-mask WHERE past its consumer, leaving the
    // renderer without its SSA value — plain `endrange` keeps the render order.
    let (k_cur, v_cur, fence) = if async_stream {
        let landed = cp_async_wait(0, smallvec![kv_idx.clone()]).barrier(smallvec![]);
        let c_k = g.cp_async_fill(&k_nxt.after(&landed), &k_l, &pf_kidx, 1);
        let c_v = g.cp_async_fill(&v_nxt.after(&landed), &v_l, &pf_kidx, 1);
        let issued: smallvec::SmallVec<[Arc<UOp>; 4]> = smallvec![landed, c_k, c_v];
        (k_cur.after(issued.clone()), v_cur.after(issued), None)
    } else {
        let s_k = g.stage_global_to_reg(&k_smem, &k_l, &pf_kidx, 1);
        let s_v = g.stage_global_to_reg(&v_smem, &v_l, &pf_kidx, 1);
        let commit_k = g.commit_reg_to_local(k_nxt, &s_k, false);
        let commit_v = g.commit_reg_to_local(v_nxt, &s_v, false);
        (k_cur, v_cur, Some([commit_k.uop().clone(), commit_v.uop().clone()]))
    };

    // Gather buf[cur] (counter-dependent ⇒ loop-scoped; reads the block landed last
    // iteration, or the prologue for block 0) and run QKᵀ → causal mask → online
    // softmax → A·V.
    let ctx = FaCtx { warp: &warp, lp: &lp, q_reg_t: &q_reg_t, q_blk: &q_blk, warpid: &warpid, causal, valid_len };
    // The two pipeline stages: gather + QKᵀ + mask, then online-softmax + A·V.
    let FaScratch { k_reg, k_reg_t, v_reg, att, att_mma, max_vec_last, att_smem } = sc;
    let (att, v_reg) = fa_qk(&ctx, k_reg, k_reg_t, v_reg, att, k_cur, v_cur, &kv_idx, fence.as_ref().map(|f| &f[..]));
    let FaAcc { norm_vec, o_reg, .. } = fa_softmax_pv(&ctx, acc, att_mma, att_smem, max_vec_last, att, &v_reg);

    let o_reg = lp.close_carry(o_reg);
    let norm_vec = norm_vec.after(&o_reg);
    // No copy may be outstanding at exit: drain the last trip's wrapped prefetch
    // before the output store (threaded through the GLOBAL tile, so the carried
    // accumulators keep their plain post-loop `After([END])` reads).
    let o = if async_stream {
        o.rewrap(o.uop().after(smallvec![cp_async_wait_all(smallvec![o_reg.uop().clone()])]))
    } else {
        o
    };

    let o_reg = o_reg / &norm_vec;
    let o_reg_t = warp.transpose(o_reg_t, &o_reg);
    let _ = warp.store(o, o_reg_t, MoveIdx::block((batch.clone(), q_blk.clone(), head.clone(), 0), 1));
}

/// Per-arch policy of [`flash_attention_with`]: the per-warp tile crossover and
/// the loop-body form. The bigger `big` tile (which amortizes the softmax over
/// more matrix-core work) is chosen once the launch grid `b·h·n/(q_blk·NUM_WARPS)`
/// covers the machine's `compute_units` and `N` divides its block; otherwise the
/// baseline `small` tile (the bigger tile shrinks the grid, so it loses at low
/// occupancy). The CU counts are the arch's flagship part (MI300X 304, Strix
/// Halo 40); `big == small` disables the crossover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaPolicy {
    pub compute_units: usize,
    pub big: (usize, usize),
    /// The largest head dim the `big` tile fits at (CUDA: `{16,64}` at d=128 needs
    /// 64 KiB of K/V double buffers, past the 48 KiB static LDS).
    pub big_max_d: usize,
    pub small: (usize, usize),
    /// Emit the flat (fully-unrolled) body. Every register index is then a
    /// constant, which the NVPTX backend needs to keep the accumulators in
    /// registers (a rolled elementwise loop it declines to unroll pins `o_reg`
    /// to local memory); the AMD path keeps the rolled iglp baseline.
    pub unroll: bool,
}

impl FaPolicy {
    /// gfx942 keeps the bench-calibrated 304-CU `{32,32}` crossover and gfx1151
    /// the baseline tile. CUDA (measured on sm_86, 28 SMs, with the `ldmatrix` +
    /// `cp.async` K/V stream): the taller KV super-block `{16,64}` (117 registers,
    /// 32 KiB LDS at d=64 — two blocks per SM) is fastest on every grid that covers
    /// the SMs (GigaAM b=8/h=16/n=1536: 3.20 ms vs 3.26 at `{16,32}` and 3.55 at
    /// `{32,32}`; whisper b=1/h=6: 192 vs 198 / 272 µs), while at d=128 its double
    /// buffers would need 64 KiB, so d=128 keeps `{16,32}` (158 registers, no
    /// spill). Both CUDA tiles are flat: the rolled body pins the register tiles to
    /// local memory (3-15× slower).
    pub fn for_arch(arch: svod_dtype::GpuArch) -> Self {
        let small = (Q_BLK, KV_BLK);
        match arch {
            svod_dtype::GpuArch::Amd(svod_dtype::AmdArch::Gfx942) => {
                Self { compute_units: 304, big: (32, 32), big_max_d: usize::MAX, small, unroll: false }
            }
            svod_dtype::GpuArch::Amd(_) => {
                Self { compute_units: 40, big: small, big_max_d: usize::MAX, small, unroll: false }
            }
            svod_dtype::GpuArch::Cuda(_) | svod_dtype::GpuArch::Metal(_) => {
                Self { compute_units: 28, big: (Q_BLK, 2 * KV_BLK), big_max_d: 64, small, unroll: true }
            }
        }
    }

    /// The `(q_blk, kv_blk)` for a `[b, n, h, d]` attention.
    pub fn tile(&self, b: usize, n: usize, h: usize, d: usize) -> (usize, usize) {
        let big_n = self.big.0 * NUM_WARPS;
        if d <= self.big_max_d && n.is_multiple_of(big_n) && b * h * (n / big_n) >= self.compute_units {
            self.big
        } else {
            self.small
        }
    }

    /// The builder config for a `[b, n, h, d]` attention.
    pub fn config(&self, b: usize, n: usize, h: usize, d: usize, causal: bool) -> FaConfig {
        let (q_blk, kv_blk) = self.tile(b, n, h, d);
        FaConfig { q_blk, kv_blk, unroll: self.unroll, causal }
    }
}

/// Run the rolled double-buffered multi-wave flash-attention forward into `o`
/// ([`build_fa_mw_rdb`]). One rolled KV loop over a parity-indexed 2× LDS double
/// buffer (one [`FaScratch`]); the per-warp tile is [`FaPolicy::tile`]. `o` is an
/// **output parameter**: the result is written in place into the supplied tensor.
///
/// ```text
/// let mut o = Tensor::empty(&[b, n, h, d], DType::BFloat16);
/// flash_attention_forward_mw_rdb(&mut o, &q, &k, &v)?;
/// // `o` now holds the attention output; read it with `o.as_vec::<bf16>()`.
/// ```
///
/// Returns `Err` if `q`/`k` aren't statically-shaped rank-4 tensors.
///
/// # Panics
/// Panics unless the head dim `D`, the per-warp `Q_BLK`/`KV_BLK` tiles, and `N`
/// satisfy the builder's divisibility asserts (`D % 16`, `Q_BLK % 16`,
/// `KV_BLK % 16`, `H % H_kv`, `N % (Q_BLK·NUM_WARPS)`).
pub fn flash_attention_forward_mw_rdb(o: &mut Tensor, q: &Tensor, k: &Tensor, v: &Tensor) -> crate::LaunchResult<()> {
    fa_check_target(q)?;
    let qd = crate::launch::concrete_dims(q, "flash-attention", "q", 4)?;
    let kd = crate::launch::concrete_dims(k, "flash-attention", "k", 4)?;
    let (b, n, h, d) = (qd[0], qd[1], qd[2], qd[3]);
    let h_kv = kd[2];
    let cfg = FaPolicy::for_arch(svod_dtype::GpuArch::Amd(svod_dtype::AmdArch::Gfx942)).config(b, n, h, d, true);
    let grid = [h as i64, (n / cfg.q_blk / NUM_WARPS) as i64, b as i64];

    let in_dtype = q.uop().dtype();
    crate::run_kernel("fa_mw_rdb", grid, (NUM_WARPS * 64) as i64, &mut [o], &[q, k, v], |ker| {
        build_fa_mw_rdb(ker, b, n, h, h_kv, d, cfg, in_dtype.clone(), false);
        ker.finish(1)
    })
}

/// Options for the unified [`flash_attention_with`] entry point.
///
/// `causal` selects the triangular (causal block-skip) sweep vs the full
/// bidirectional sweep. `key_lens` is an optional realized `[B]`-shaped `i32`
/// tensor of valid **key** counts per batch — a *key-only* padding mask: keys at
/// `kv_pos >= key_lens[batch]` are masked out of every query row. Queries beyond
/// the valid length are still computed (the kernel does not mask query rows); the
/// caller is expected to discard those padded output rows. The scheduler fallback
/// mirrors this exactly with a `[B,1,1,N]` key mask, so the hand kernel and the
/// fallback agree on every row (valid and padded alike).
#[derive(Clone, Copy)]
pub struct FaOpts<'a> {
    /// Causal (triangular) attention when `true`; full bidirectional when `false`.
    pub causal: bool,
    /// Optional `[B]` `i32` per-batch valid-key-count padding mask (key-only).
    pub key_lens: Option<&'a Tensor>,
}

impl Default for FaOpts<'_> {
    fn default() -> Self {
        Self { causal: true, key_lens: None }
    }
}

/// **Graph-native** flash-attention forward — runs the hand kernel, or reports
/// that it doesn't apply. **No silent fallback:** the caller owns that policy.
///
/// Q is `[B,N,H,D]`, K/V are `[B,N,H_KV,D]`. The outcome is three-way, splitting
/// "this device/length can't use the kernel" (`None`, a fallback trigger) from
/// "this request is malformed" (`Err`, a caller bug):
///
/// - `Ok(Some(out))` — ran: a lazy output [`Tensor`] (`custom_kernel` / `Op::Call`
///   node) from the rolled double-buffered kernel ([`build_fa_mw_rdb`]) via
///   [`crate::graph_launch`], honoring `opts.causal` and the optional
///   `opts.key_lens` **key-only** mask (a 5th `[B]` `i32` global after `o,q,k,v`).
/// - `Ok(None)` — *doesn't apply here:* the device isn't a supported arch
///   ([`FA_SUPPORTED_ARCHS`] — gfx942/gfx1151/CUDA sm_80+ with its LLVM backend), **or** the
///   runtime sequence length doesn't tile (`N % (q_blk·NUM_WARPS) != 0`). The caller
///   substitutes its own attention (e.g. [`Tensor::scaled_dot_product_attention`]).
/// - `Err` — *malformed request* on a supported device: a FIXED property is wrong —
///   `q`/`k` not a statically-shaped rank-4 tensor, operand dtype ∉ {bf16, f16},
///   `D % 16 != 0`, or `H % H_KV != 0` (GQA). These are
///   caller bugs, raised loudly instead of silently routed to the slow path. (A
///   genuine kernel build/dispatch failure also returns `Err`.)
///
/// ```no_run
/// use svod_tensor::Tensor;
/// use svod_dtype::DType;
/// use svod_tk::FaOpts;
/// let q = Tensor::randn(&[1, 128, 16, 64]).unwrap().cast(DType::BFloat16).unwrap();
/// let (k, v) = (q.clone(), q.clone());
/// // `None` ⇒ the kernel doesn't apply here; the caller picks the fallback.
/// if let Some(mut o) = svod_tk::flash_attention_with(&q, &k, &v, FaOpts { causal: false, key_lens: None }).unwrap() {
///     o.prepare().unwrap();
/// }
/// ```
pub fn flash_attention_with(q: &Tensor, k: &Tensor, v: &Tensor, opts: FaOpts) -> crate::LaunchResult<Option<Tensor>> {
    let qd = crate::launch::concrete_dims(q, "flash-attention", "q", 4)?;
    let kd = crate::launch::concrete_dims(k, "flash-attention", "k", 4)?;
    let (b, n, h, d) = (qd[0], qd[1], qd[2], qd[3]);
    let h_kv = kd[2];
    let dtype = q.uop().dtype();
    let dtype_ok = dtype == DType::BFloat16 || dtype == DType::Float16;
    let err_dtype = dtype.clone();

    crate::launch_custom(
        &q.device(),
        FA_SUPPORTED_ARCHS,
        // Structural validity (`Err`) — operand dtype, head dim, and GQA divisibility
        // are FIXED model properties; a violation on a supported device is a caller bug.
        move |_arch| {
            ensure!(
                dtype_ok,
                crate::launch::DtypeSnafu { kernel: "flash-attention", got: err_dtype, expected: "bf16 or f16" }
            );
            ensure!(
                d % BLK == 0,
                crate::launch::DimMultipleSnafu {
                    kernel: "flash-attention",
                    dim: "head dim D",
                    value: d,
                    multiple: BLK
                }
            );
            ensure!(
                h % h_kv == 0,
                crate::launch::DimDivisibleSnafu {
                    kernel: "flash-attention",
                    dim: "H",
                    value: h,
                    divisor: "H_kv",
                    divisor_value: h_kv,
                }
            );
            Ok(())
        },
        // Runtime tiling (`None`) — `N` is the (audio) sequence length and may
        // legitimately not tile, so the caller falls back per-clip instead of padding.
        move |arch| n % (FaPolicy::for_arch(arch).tile(b, n, h, d).0 * NUM_WARPS) == 0,
        // Build for the resolved arch — caps track the real wave width.
        move |arch| {
            let caps = crate::ArchCaps::for_arch(arch);
            let cfg = FaPolicy::for_arch(arch).config(b, n, h, d, opts.causal);
            let grid = [h as i64, (n / cfg.q_blk / NUM_WARPS) as i64, b as i64];
            let out = Tensor::empty(&[b, n, h, d], dtype.clone());
            let masked = opts.key_lens.is_some();
            let build_dtype = dtype.clone();
            // ABI/global order is o, q, k, v, (lens) — `out` is global[0], inputs map to
            // global[1..] in order, so `key_lens` (the 5th global) goes last.
            //
            // Clamp key_lens to >= 1. A fully key-masked row (key_lens[b] == 0, an
            // inactive zero-padded lane) has no valid key, so the online-softmax
            // running max stays -inf and the rescale's -inf - (-inf) is NaN that
            // poisons the row. Flooring to >= 1 makes every row attend to at least
            // key 0 (a finite value) — reducing the degenerate case to the ordinary
            // partial-mask path. Such inactive lanes are caller-discarded, so the
            // exact value is immaterial (only finiteness is); partial masks (already
            // >= 1 valid key) are unchanged.
            //
            // The clamp is a property of `key_lens`, not of the calling layer: every
            // layer sharing one `key_lens` must share one clamp kernel, so it is
            // built outside the caller's origin scope.
            let key_lens_clamped = opts.key_lens.map(|lens| {
                let _shared = svod_ir::origin::OriginScope::suspend();
                let ones = Tensor::full(&[b], ConstValue::Int(1), DType::Int32).expect("ones[b]");
                lens.maximum(&ones).expect("clamp key_lens >= 1")
            });
            let mut ins: Vec<&Tensor> = vec![q, k, v];
            if let Some(lens) = &key_lens_clamped {
                ins.push(lens);
            }
            let block = (NUM_WARPS * caps.wave_size) as i64;
            crate::graph_launch("flash_attention", grid, block, out, &ins, caps, move |ker| {
                build_fa_mw_rdb(ker, b, n, h, h_kv, d, cfg, build_dtype.clone(), masked);
                ker.finish(1)
            })
        },
    )
}

/// **Graph-native** causal flash-attention forward — thin wrapper over
/// [`flash_attention_with`] with [`FaOpts::default`] (causal, unmasked). Returns
/// `Ok(Some(out))` (a lazy `custom_kernel` / `Op::Call` [`Tensor`]) when the kernel
/// applies, `Ok(None)` otherwise; see [`flash_attention_with`] for the eligibility
/// rules and the no-silent-fallback contract.
///
/// ```no_run
/// use svod_tensor::Tensor;
/// use svod_dtype::DType;
/// let q = Tensor::randn(&[1, 128, 16, 64]).unwrap().cast(DType::BFloat16).unwrap();
/// let (k, v) = (q.clone(), q.clone());
/// if let Some(mut o) = svod_tk::flash_attention(&q, &k, &v).unwrap() {
///     o.prepare().unwrap();
/// }
/// ```
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> crate::LaunchResult<Option<Tensor>> {
    flash_attention_with(q, k, v, FaOpts::default())
}
