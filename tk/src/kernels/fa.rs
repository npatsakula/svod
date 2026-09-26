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

use std::cell::OnceCell;
use std::rc::Rc;
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
/// → [`build_fa_mw_rdb`]) is enabled for: gfx942 (CDNA MFMA, wave64), the wave32
/// RDNA parts (gfx1151 gfx11 WMMA, gfx1200/gfx1201 RDNA4 WMMA), CUDA sm_80+
/// (`mma.sync`, warp32) and Apple7+
/// (`simdgroup_matrix`, SIMD-group 32). The launcher gates
/// against this list; generic launch infrastructure stays architecture-agnostic.
/// gfx942 was validated on hardware before the vector LDS gathers and the single
/// fenced K/V commit (PR #177) and has not been re-run since; its golden graph
/// digests were re-baselined for those two changes without it.
pub const FA_SUPPORTED_ARCHS: crate::ArchSet = crate::ArchSet::amd(crate::target::CDNA_RDNA_WMMA)
    .with_cuda_from(svod_dtype::CudaArch::from_compute_capability(8, 0))
    .with_metal_from(svod_dtype::MetalFamily::Apple(7));

/// Whether `device` can run the production graph flash-attention kernel.
/// Uses the same architecture and toolchain gate as [`crate::launch_custom`].
pub fn flash_attention_supported(device: &svod_dtype::DeviceSpec) -> bool {
    crate::target::resolve_supported_arch(device, FA_SUPPORTED_ARCHS).is_ok()
}

/// The **direct-launch** FA wrapper ([`flash_attention_forward_mw_rdb`]) builds the
/// wave64 block and the CDNA fragment tiles, so it stays gfx942-only.
const FA_DIRECT_SUPPORTED_ARCHS: crate::ArchSet = crate::ArchSet::amd(&[svod_dtype::AmdArch::Gfx942]);

/// Validate a direct-launch wrapper's device against [`FA_DIRECT_SUPPORTED_ARCHS`].
fn fa_check_target(t: &Tensor) -> crate::LaunchResult<()> {
    crate::target::check_target(&t.device(), FA_DIRECT_SUPPORTED_ARCHS)
}

/// Tuning knobs for [`build_fa_mw_rdb`] — the structured replacement for its former
/// positional `bool`/tile args (mirrors [`crate::kernels::gemm::MatmulCfg`]).
/// [`Default`] is the production baseline: `{16,16}` per-warp tile, rolled (looped)
/// causal compute. The shape (`b,n,h,h_kv,d`) stays a positional arg since it's
/// derived from the input tensors, not a tuning choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    /// Sliding-window band `(left, right)`: query `q` sees keys `q - left ..= q + right`,
    /// and each workgroup sweeps only the KV super-blocks its band touches.
    pub window: Option<(usize, usize)>,
}

impl Default for FaConfig {
    fn default() -> Self {
        Self { q_blk: Q_BLK, kv_blk: KV_BLK, unroll: false, causal: true, window: None }
    }
}

/// The optional score masks a build binds, each a trailing global after
/// `o, q, k, v` in this order: the `[B]` valid-key counts of
/// [`FaOpts::key_lens`], then the `[B, N]` segment starts of
/// [`FaOpts::seg_start`], then the `[B, N]` key validity of [`FaOpts::key_mask`]
/// (as an f32 bias where it seeds the scores, [`key_mask_seeds`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaMask {
    pub key_lens: bool,
    pub seg_start: bool,
    pub key_mask: bool,
}

impl FaMask {
    pub const NONE: Self = Self { key_lens: false, seg_start: false, key_mask: false };

    /// The masks as a bit set, the tune key's shape component.
    pub const fn code(self) -> usize {
        self.key_lens as usize | (self.seg_start as usize) << 1 | (self.key_mask as usize) << 2
    }
}

/// Where a build reads Q, K and V from, bound right after `o`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FaOperands {
    /// Three globals: Q `[B, N, H, D]`, K and V `[B, N, H_kv, D]`.
    #[default]
    Split,
    /// One `[B, N, H + 2·H_kv, D]` global — a fused QKV projection's output read
    /// by head: Q's heads, then K's, then V's. A block's rows then lie
    /// `H + 2·H_kv` heads apart instead of `H_kv`, which the 16-byte copies do
    /// not notice, and the projection's output needs no split into three
    /// tensors first.
    Packed,
}

impl FaOperands {
    /// The kernel's name, which keys its tuning apart from the other layout's.
    const fn kernel(self) -> &'static str {
        match self {
            FaOperands::Split => "flash_attention",
            FaOperands::Packed => "flash_attention_packed",
        }
    }

    /// The input globals' head counts, in binding order.
    fn heads(self, h: usize, h_kv: usize) -> Vec<usize> {
        match self {
            FaOperands::Split => vec![h, h_kv, h_kv],
            FaOperands::Packed => vec![h + 2 * h_kv],
        }
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
    /// Per-warp LDS scratch (`[NUM_WARPS·kv_blk, q_blk]`) for the
    /// `att → att_mma` accumulator→input relayout. `None` wherever
    /// [`ArchCaps::acc_reusable_as_input`](crate::ArchCaps::acc_reusable_as_input)
    /// holds (gfx942, CUDA, Metal) so a register `copy` suffices; `Some` on RDNA,
    /// where the relayout must round-trip through LDS — on gfx11 because the even/odd
    /// accumulator and the replicated input genuinely differ.
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
    window: Option<(usize, usize)>,
    valid_len: Option<Arc<UOp>>,
    /// The `[B, N]` segment-start table and the batch index it is read at.
    seg_start: Option<(crate::tile::GL, Arc<UOp>)>,
    /// The `[B, N]` key-validity table and the batch index it is read at.
    key_mask: Option<(crate::tile::GL, Arc<UOp>)>,
    /// The key mask is the f32 bias the `QKᵀ` accumulator starts from
    /// ([`key_mask_seeds`]), not an `i32` table the scores are masked by.
    seed_key_mask: bool,
    /// `log2(e)/sqrt(d)` — the softmax scale, folded with the `exp2` base change.
    /// Applied to the f32 `QKᵀ` accumulator rather than to `Q`: scaling `Q` costs a
    /// second rounding to the 16-bit mma input dtype, and that error enters the
    /// scores relative to their own magnitude, which `exp2` then amplifies. Scores
    /// grow with the square of the activation scale, so pre-scaling `Q` is accurate
    /// only near unit variance and drifts badly on real activations.
    score_scale: f64,
}

/// Whether the `[B, N]` key mask reaches the kernel as an f32 bias (`0` for a
/// valid key, `−∞` for a hidden one) that the `QKᵀ` accumulator starts from,
/// rather than as an `i32` validity table the scores are masked by after it. A
/// key's validity is one value per column, so the seed costs a load per key a
/// lane holds and nothing per score: at 1×8192 on sm_86 the masked kernel runs
/// within 1% of the unmasked one instead of 3% behind it; on gfx1201, where the
/// table's compare and select cost registers too, 2796 µs instead of 3229. The
/// scores are bit-identical — `0 + s = s`, and a hidden key stays `−∞` through
/// the MMA and the scale. Validated on CUDA and gfx12 only.
fn key_mask_seeds(caps: &crate::ArchCaps) -> bool {
    caps.cuda().is_some() || caps.amd().is_some_and(svod_dtype::AmdArch::is_rdna4)
}

/// The dtype the kernel reads the key mask in ([`key_mask_seeds`]).
fn key_mask_dtype(caps: &crate::ArchCaps) -> DType {
    if key_mask_seeds(caps) { DType::Float32 } else { DType::Int32 }
}

/// A `[B, N]` key validity (any integer or bool dtype, non-zero = valid) as the
/// kernel reads it ([`key_mask_seeds`]).
fn key_mask_operand(valid: &Tensor, caps: &crate::ArchCaps) -> Tensor {
    if !key_mask_seeds(caps) {
        return valid.cast(DType::Int32);
    }
    let shape = [valid.dim_const(0).expect("static key mask"), valid.dim_const(1).expect("static key mask")];
    let bias = |v: f64| Tensor::full(&shape, ConstValue::Float(v), DType::Float32);
    bias(0.0).where_(&valid.cast(DType::Bool), bias(f64::NEG_INFINITY)).expect("key-mask bias")
}

/// Apply the FA score-mask (causal + optional padding + optional segments +
/// optional window + optional key mask) to the `att` tile. The causal mask zeros
/// (via `−∞`) keys ahead of this warp's own query rows (`kv_pos > q_pos`); the
/// padding mask zeros keys at/after the per-batch valid length
/// (`kv_pos >= valid_len`); the segment mask zeros keys before the query's own
/// segment (`kv_pos < seg_start[batch, q_pos]`); the window zeros keys outside
/// `q_pos - left ..= q_pos + right`; the key mask zeros keys whose
/// `key_mask[batch, kv_pos]` is `0`. With none, the tile is returned unchanged
/// (the early-return avoids emitting any mask IR when masking is off). The
/// per-element `(kv_pos, q_pos)` is computed arch-correctly inside
/// [`Group::mask_where`].
fn score_mask<'k>(ctx: &FaCtx<'_, 'k>, att: RT<'k>, slice_idx: &Arc<UOp>) -> RT<'k> {
    /// `(kv_pos, q_pos) → hidden`.
    type Hidden<'a> = &'a dyn Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>;
    let FaCtx { warp, causal, window, .. } = *ctx;
    let key_mask = ctx.key_mask.as_ref().filter(|_| !ctx.seed_key_mask);
    let (valid_len, seg_start) = (ctx.valid_len.as_ref(), ctx.seg_start.as_ref());
    if !causal && valid_len.is_none() && seg_start.is_none() && window.is_none() && key_mask.is_none() {
        return att;
    }
    let row_blk = Idx::Uop(slice_idx.clone());
    let col_blk = Idx::Uop(ctx.q_blk.clone());
    let mask = |att, predicate: Hidden<'_>| {
        warp.mask_where(att, row_blk.clone(), col_blk.clone(), f64::NEG_INFINITY, predicate)
    };
    let att = if causal { mask(att, &|kv_pos, q_pos| kv_pos.gt(q_pos)) } else { att };
    let att = if let Some(vl) = valid_len { mask(att, &|kv_pos, _| kv_pos.ge(vl)) } else { att };
    let att = if let Some((table, batch)) = seg_start {
        mask(att, &|kv_pos, q_pos| {
            kv_pos.lt(&load_at(table.uop(), table.shape(), &[Idx::from(batch), Idx::from(q_pos)]))
        })
    } else {
        att
    };
    let att = if let Some((left, right)) = window {
        let att = mask(att, &|kv_pos, q_pos| kv_pos.add(&iconst(left as i64)).lt(q_pos));
        mask(att, &|kv_pos, q_pos| kv_pos.gt(&q_pos.add(&iconst(right as i64))))
    } else {
        att
    };
    if let Some((table, batch)) = key_mask {
        mask(att, &|kv_pos, _| {
            load_at(table.uop(), table.shape(), &[Idx::from(batch), Idx::from(kv_pos)])
                .eq(&UOp::const_(DType::Int32, ConstValue::Int(0)))
        })
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

    // QKᵀ into a freshly-seeded att tile (re-seeded each trip via the loop scope):
    // zeros, or the key mask's `0`/`−∞` bias ([`key_mask_seeds`]).
    let att = ctx.lp.reinit(att);
    let att = match ctx.key_mask.as_ref().filter(|_| ctx.seed_key_mask) {
        Some((table, batch)) => {
            let (kv_blk, q_blk) = (Idx::Uop(slice_idx.clone()), Idx::Uop(ctx.q_blk.clone()));
            warp.map_position(att, kv_blk, q_blk, |_, _, kv_pos, _| {
                load_at(table.uop(), table.shape(), &[Idx::from(batch), Idx::from(kv_pos)])
            })
        }
        None => warp.zero(att),
    };
    let k_reg_t = warp.transpose(k_reg_t, &k_reg);
    let att = warp.mma_atb(att, &k_reg_t, ctx.q_reg_t);
    // Scale in f32, on the accumulator — see `FaCtx::score_scale`.
    let att = att * ctx.score_scale;

    let att = score_mask(ctx, att, slice_idx);
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
    max_vec = warp.col_reduce(max_vec.after(&max_vec_last), &att, |a, b| warp.max_num(a, b), f64::NEG_INFINITY);

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
    // store the accumulator (matrix `(kv,q)` order), barrier, reload under the input
    // map (on gfx11 `K=kv=element`, `N=q=lane%16`). Both lane maps are the
    // matmul-validated ones, so the relayout is correct by construction.
    // `att` narrows as if never NaN ([`Group::narrow_finite`]): a NaN score
    // still reaches the output through `norm_vec`, summed from the f32 `att`.
    let att_mma = match att_smem {
        None => warp.narrow_finite(att_mma.after((lp.index(), &norm_vec)), &att),
        Some(att_smem) => {
            // This warp's `(kv_blk × q_blk)` band of the shared relayout buffer, as a
            // zero-copy subtile — so the store and the reload address the warp's band
            // with no repeated wave-block index (mirrors the matmul LDS gather). The
            // band size is `att`'s element shape (its fragment grid × the base edge).
            let an = att.shape().len();
            let dims = (att.shape()[an - 3] * att.base.base.rows, att.shape()[an - 2] * att.base.base.cols);
            let band = att_smem.subtile(dims, (ctx.warpid.clone(), 0));
            let deps = smallvec![lp.index().clone(), norm_vec.uop().clone()];
            let stored = warp.store_local_fenced(band, &att, MoveIdx::default(), deps);
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
    mask: FaMask,
) {
    build_fa(ker, (b, n, h, h_kv, d), cfg, in_dtype, mask, FaOperands::Split);
}

/// [`build_fa_mw_rdb`] with Q, K and V bound per `operands`.
fn build_fa(
    ker: &Kernel,
    (b, n, h, h_kv, d): (usize, usize, usize, usize, usize),
    cfg: FaConfig,
    in_dtype: DType,
    mask: FaMask,
    operands: FaOperands,
) {
    let FaConfig { q_blk: q_blk_rows, kv_blk: kv_blk_rows, unroll, causal, window } = cfg;
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

    // ABI: outputs (o) then inputs (q, k, v — or the one packed qkv), fixed by
    // construction.
    let in_specs: Vec<GlSpec> =
        operands.heads(h, h_kv).into_iter().map(|heads| GlSpec::new(&[b, n, heads, d], in_dtype.clone())).collect();
    let (outs, ins) = ker.bind_abi(&[GlSpec::new(&[b, n, h, d], in_dtype.clone())], &in_specs);
    let (o, q) = (outs[0].clone(), ins[0].clone());
    let (k, v) = match operands {
        FaOperands::Split => (ins[1].clone(), ins[2].clone()),
        FaOperands::Packed => (q.clone(), q.clone()),
    };
    // Per-batch valid key-length buffer (padding mask), bound AFTER o,q,k,v (trailing —
    // never interleaved) so the ABI slot order stays stable; only bound when `masked`.
    // The scalar `lens[batch]` is already int32, matching the concrete SPECIAL
    // position arithmetic.
    let valid_len = mask.key_lens.then(|| {
        let lens = ker.gl(&[b], DType::Int32);
        load_at(lens.uop(), lens.shape(), &[Idx::from(&ker.block_idx[2])])
    });
    // The `[B, N]` segment starts, bound after `lens`; read per query row inside
    // the score mask at this workgroup's batch.
    let seg_start = mask.seg_start.then(|| (ker.gl(&[b, n], DType::Int32), ker.block_idx[2].clone()));
    // The `[B, N]` key validity, bound last: the scores' seed, or read per key
    // inside the score mask ([`key_mask_seeds`]).
    let key_mask = mask.key_mask.then(|| (ker.gl(&[b, n], key_mask_dtype(&ker.caps)), ker.block_idx[2].clone()));

    let head = ker.grid_x();
    let head_kv = head.floor_div(&iconst(group_size));
    // K's and V's heads in their globals: packed, they follow Q's `h` heads.
    let (k_head, v_head) = match operands {
        FaOperands::Split => (head_kv.clone(), head_kv.clone()),
        FaOperands::Packed => (head_kv.add(&iconst(h as i64)), head_kv.add(&iconst((h + h_kv) as i64))),
    };
    let batch = ker.grid_z();
    let block_q_base = ker.grid_y();
    let warpid = g.warpid_in_group();
    let q_blk = block_q_base.mul(&iconst(NUM_WARPS as i64)).add(&warpid);

    let in_dt = in_dtype.clone();
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
    // Q feeds the B operand of the QKᵀ mma (K feeds A, V feeds A of the PV mma), so
    // its gather takes the B-position fragment map — a no-op off Metal, where A and B
    // share one fragment; see `FragRole::Operand`.
    let q_reg = ker.operand_b((q_blk_rows, d), in_dt.clone(), row);
    let q_reg_t = ker.operand_b((d, q_blk_rows), in_dt.clone(), col);
    let o_reg_t = ker.acc_t((q_blk_rows, d), row);

    // One scratch set: the rolled body has a back-edge, so the carried FaAcc + a
    // single set suffice. `att_smem` holds one per-warp relayout band on RDNA.
    let sc = FaScratch {
        k_reg: ker.operand((kv_blk_rows, d), in_dt.clone(), row),
        k_reg_t: ker.operand((d, kv_blk_rows), in_dt.clone(), col),
        v_reg: ker.operand((kv_blk_rows, d), in_dt.clone(), col),
        att: ker.acc((kv_blk_rows, q_blk_rows), col),
        att_mma: ker.operand_b((kv_blk_rows, q_blk_rows), in_dt.clone(), col),
        max_vec_last: ker.acc_vec(q_blk_rows),
        att_smem: (!ker.caps.acc_reusable_as_input())
            .then(|| ker.shared((NUM_WARPS * kv_blk_rows, q_blk_rows), in_dt.clone(), row)),
    };

    // Carried online-softmax accumulators.
    let o_reg = ker.acc((d, q_blk_rows), col);
    let max_vec = ker.acc_vec(q_blk_rows);
    let norm_vec = ker.acc_vec(q_blk_rows);
    // The running max starts at the finite floor, not `-∞`: a KV block the
    // masks hide entirely from a query row (a packed row's earlier segments)
    // leaves the max where it was, and the rescale `exp2(prev - new)` needs a
    // finite `prev` to be `1`, not `-∞ - (-∞) = NaN`. Against real scores the
    // floor rescales to `exp2(-3.4e38) = 0` exactly as `-∞` did.
    let max_floor = warp.clear_rv(max_vec, f64::from(f32::MIN));
    let acc = FaAcc { max_vec: max_floor, norm_vec: warp.zero_rv(norm_vec), o_reg: warp.zero(o_reg) };

    // Load this warp's Q tile, then transpose for the QKᵀ contraction. The gather
    // lands the 16-bit operand dtype straight in registers: the softmax scale rides
    // on the f32 accumulator instead (`FaCtx::score_scale`), so there is nothing to
    // scale here and an f32 staging tile would only cast the stored value out and
    // back (64 spare f32 registers per lane for a round trip).
    let q_reg = warp.load(q_reg, q, MoveIdx::block((batch.clone(), q_blk.clone(), head.clone(), 0), 1));
    let q_reg_t = warp.transpose(q_reg_t, &q_reg);

    // Total KV super-blocks (the full bidirectional sweep). With `causal`, the
    // per-q-block bound is the causal block-skip `(block_q_base+1)*NUM_WARPS*Q_BLK/KV_BLK`
    // super-blocks; without it every q-block attends to all `total_kv_blocks`. A
    // window starts the sweep at the first super-block its band touches and ends it
    // at the last (`kv_start` + `kv_bound` trips), so a local layer reads
    // `(q_rows + left + right) / KV_BLK` blocks instead of all of them.
    let total_kv_blocks = (n / kv_blk_rows) as i64;
    let blocks_mult = (NUM_WARPS * q_blk_rows / kv_blk_rows) as i64;
    let causal_end = || block_q_base.add(&iconst(1)).mul(&iconst(blocks_mult));
    let (kv_start, kv_bound) = match window {
        None if causal => (None, causal_end()),
        None => (None, iconst(total_kv_blocks)),
        Some((left, right)) => {
            let (left, right, kv) = (left as i64, right as i64, kv_blk_rows as i64);
            let q_rows = blocks_mult * kv;
            let q0 = block_q_base.mul(&iconst(q_rows));
            // `max(q0 - left, 0)` and `min(a, b) = a - max(a - b, 0)`: every
            // intermediate stays non-negative, so floor and truncating division agree.
            let start = q0.max(&iconst(left)).sub(&iconst(left)).floor_div(&iconst(kv));
            let min = |a: Arc<UOp>, b: Arc<UOp>| a.sub(&a.sub(&b).max(&iconst(0)));
            let end = q0.add(&iconst(q_rows + right + kv - 1)).floor_div(&iconst(kv));
            let end = min(end, iconst(total_kv_blocks));
            let end = if causal { min(end, causal_end()) } else { end };
            let trips = end.sub(&start);
            (Some(start), trips)
        }
    };

    // The K/V stream: `cp.async` (sm_80+ — the copy lands in LDS with no register
    // staging and stays in flight under the previous block's compute) or the
    // register-staged prefetch (AMD).
    let async_stream = g.cp_async_fill_applies(&k_smem, &k) && g.cp_async_fill_applies(&v_smem, &v);

    // Prologue: the sweep's first block → buf[0]. Register-staged: stage → VGPR, commit, barrier;
    // cp.async: issue + commit only (the loop top retires and fences it).
    let first_blk = kv_start.as_ref().map_or(Idx::Const(0), Idx::from);
    let p_kidx = [Idx::from(&batch), first_blk.clone(), Idx::from(&k_head), Idx::Const(0)];
    let p_vidx = [Idx::from(&batch), first_blk, Idx::from(&v_head), Idx::Const(0)];
    let (k_smem, v_smem) = if async_stream {
        let c_k = g.cp_async_fill(&k_smem, &k, &p_kidx, 1);
        let c_v = g.cp_async_fill(&v_smem, &v, &p_vidx, 1);
        (k_smem.after(c_k), v_smem.after(c_v))
    } else {
        let s0_k = g.stage_global_to_reg(&k_smem, &k, &p_kidx, 1);
        let s0_v = g.stage_global_to_reg(&v_smem, &v, &p_vidx, 1);
        let landed = g.commit_regs_to_local(&[(&k_smem, &s0_k), (&v_smem, &s0_v)]).barrier(smallvec![]);
        ker.push_store(landed.clone(), k_smem.uop().clone());
        (k_smem.after(&landed), v_smem.after(&landed))
    };

    // Rolled KV loop. `kv_bound` (the dynamic per-q-block causal trip count) is the
    // Range end. The prefetch-block index is `(kv+1) % total_kv_blocks` (a FloorMod): the
    // final trip's prefetch (`kv+1 == total`) wraps to block 0, which is never
    // gathered, keeping the GLOBAL read in bounds. A `min`/`where` clamp is avoided
    // — a `WHERE` in the prefetch-address path is mis-ordered past its address-MUL
    // consumer in this kernel's linearization, leaving the renderer without its SSA
    // value; FloorMod (like the parity) lowers and orders cleanly.
    //
    // ONE tracked loop: splitting the causal sweep into an unmasked phase plus a
    // masked diagonal phase (which would lift the mask off ~60% of the trips, worth
    // ~3% here) needs two tracked ranges in one kernel, and the kernel-graph former
    // then declines to wrap the body as an opaque CALL (its terminal `END(STORE)`
    // lands in the outer graph and fails `spec_kernel_graph`). Measured, not assumed.
    let lp = ker.loop_dynamic(kv_bound);
    let kv_idx = lp.index().clone();
    let kvp1 = kv_idx.add(&iconst(1));
    // The super-block this trip reads and the one it prefetches; the trip index
    // alone picks the buffer half.
    let blk = kv_start.as_ref().map_or_else(|| kv_idx.clone(), |s| s.add(&kv_idx));
    let pf = kv_start.as_ref().map_or_else(|| kvp1.clone(), |s| s.add(&kvp1));
    let pf = pf.try_mod(&iconst(total_kv_blocks)).expect("(kv+1) % total blocks");
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
    let pf_kidx = [Idx::from(&batch), Idx::from(&pf), Idx::from(&k_head), Idx::Const(0)];
    let pf_vidx = [Idx::from(&batch), Idx::from(&pf), Idx::from(&v_head), Idx::Const(0)];
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
    // per-commit barrier: the store node is handed to the WAR barrier the gathers
    // consume, which gates both the cross-iteration RAW and WAR), gather buf[cur]. The barrier-wrapped END (`endrange_barrier_to`)
    // is NOT used: it reorders the causal-mask WHERE past its consumer, leaving the
    // renderer without its SSA value — plain `endrange` keeps the render order.
    let (k_cur, v_cur, fence) = if async_stream {
        let landed = cp_async_wait(0, smallvec![kv_idx.clone()]).barrier(smallvec![]);
        let c_k = g.cp_async_fill(&k_nxt.after(&landed), &k_l, &pf_kidx, 1);
        let c_v = g.cp_async_fill(&v_nxt.after(&landed), &v_l, &pf_vidx, 1);
        let issued: smallvec::SmallVec<[Arc<UOp>; 4]> = smallvec![landed, c_k, c_v];
        (k_cur.after(issued.clone()), v_cur.after(issued), None)
    } else {
        let s_k = g.stage_global_to_reg(&k_smem, &k_l, &pf_kidx, 1);
        let s_v = g.stage_global_to_reg(&v_smem, &v_l, &pf_vidx, 1);
        // One store node for both strips; the gathers' WAR fence below is the
        // barrier that covers it.
        let committed = g.commit_regs_to_local(&[(&k_nxt, &s_k), (&v_nxt, &s_v)]);
        (k_cur, v_cur, Some([committed]))
    };

    // Gather buf[cur] (counter-dependent ⇒ loop-scoped; reads the block landed last
    // iteration, or the prologue for block 0) and run QKᵀ → causal mask → online
    // softmax → A·V.
    let score_scale = (1.0 / (d as f64).sqrt()) * std::f64::consts::LOG2_E;
    let ctx = FaCtx {
        warp: &warp,
        lp: &lp,
        q_reg_t: &q_reg_t,
        q_blk: &q_blk,
        warpid: &warpid,
        causal,
        window,
        valid_len,
        seg_start,
        key_mask,
        seed_key_mask: key_mask_seeds(&ker.caps),
        score_scale,
    };
    // The two pipeline stages: gather + QKᵀ + mask, then online-softmax + A·V.
    let FaScratch { k_reg, k_reg_t, v_reg, att, att_mma, max_vec_last, att_smem } = sc;
    let (att, v_reg) = fa_qk(&ctx, k_reg, k_reg_t, v_reg, att, k_cur, v_cur, &blk, fence.as_ref().map(|f| &f[..]));
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

    // A row the masks hide entirely (a padded query under a window, or a key mask
    // with no valid key) sums to 0: floor its norm so it stores exact zeros, as
    // SDPA does, instead of `0/0`. A row that sees any key sums to at least 1.
    let norm_vec = if mask.key_mask || window.is_some() {
        norm_vec.maximum(&warp.clear_rv(ker.acc_vec(q_blk_rows), f64::from(f32::MIN_POSITIVE)))
    } else {
        norm_vec
    };
    let o_reg = o_reg / &norm_vec;
    let o_reg_t = warp.transpose(o_reg_t, &o_reg);
    let _ = warp.store(o, o_reg_t, MoveIdx::block((batch.clone(), q_blk.clone(), head.clone(), 0), 1));
}

/// Per-arch policy of [`flash_attention_with`]: the per-warp tile crossover and
/// the loop-body form. A [`Self::big`] tile is chosen once the launch grid
/// `b·h·n/(q_blk·NUM_WARPS)` covers the device's `compute_units` and `N` divides
/// its block; otherwise the baseline `small` tile, since a grid that does not
/// cover the device wants the tile that amortizes the softmax over the most
/// matrix-core work rather than the one that fits the most blocks per CU.
/// [`Self::for_device`] reads the CU count off the device; [`Self::for_arch`]
/// alone assumes the arch's flagship part (MI300X 304, Strix Halo 40, an RTX
/// 3060's 28 SMs). A `big` entry equal to `small` disables the crossover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaPolicy {
    pub compute_units: usize,
    /// The grid-covering tiles as `(head-dim bound, tile)` pairs in increasing
    /// bound order — the first pair whose bound covers `d` supplies the tile, and
    /// a `d` past every bound falls back to `small`. The right tile is head-dim
    /// dependent because the per-warp register tiles scale with `d`: the taller
    /// KV super-block wins while the block still fits twice on a CU, and loses to
    /// the square tile once it does not.
    pub big: &'static [(usize, (usize, usize))],
    pub small: (usize, usize),
    /// Emit the flat (fully-unrolled) body. Every register index is then a
    /// constant, which the NVPTX backend needs to keep the accumulators in
    /// registers (a rolled elementwise loop it declines to unroll pins `o_reg`
    /// to local memory); the AMD path keeps the rolled iglp baseline.
    pub unroll: bool,
    /// Shared memory a block may take: the static 48 KiB on CUDA, the 64 KiB
    /// LDS on AMD. A tile whose buffers exceed it is not chosen, so a large
    /// head dim declines (`None`) instead of failing in the assembler.
    pub shared_max: usize,
    /// Whether the body stages the softmax band through shared memory (RDNA,
    /// where the accumulator is not reusable as an operand).
    pub att_band: bool,
}

/// Bytes per element of the 16-bit operand dtypes the kernel accepts.
const IN_BYTES: usize = 2;

/// The per-warp tiles [`FaPolicy::tuned`] measures on first use: the table
/// entries every policy draws from, in increasing register footprint.
pub const FA_TILES: [(usize, usize); 4] =
    [(Q_BLK, Q_BLK), (Q_BLK, KV_BLK), (Q_BLK, 2 * KV_BLK), (2 * Q_BLK, 2 * Q_BLK)];

impl FaPolicy {
    /// The measured crossovers: `{32,32}` on CDNA (gfx942); on RDNA (gfx1151)
    /// `{16,32}` at d ≤ 64 and `{16,16}` at d = 128, where the wider KV block
    /// costs occupancy (b=8/h=16/n=512 causal: 1.02 vs 1.64 ms); on CUDA (sm_86)
    /// `{16,64}` at d ≤ 64 and `{16,16}` at d = 128 — at each head dim the widest
    /// tile that still fits two blocks per SM. The CUDA bodies must stay flat: a
    /// rolled body pins the register tiles to local memory (3-15× slower).
    pub fn for_arch(arch: svod_dtype::GpuArch) -> Self {
        let small = (Q_BLK, KV_BLK);
        let att_band = !crate::ArchCaps::for_arch(arch).acc_reusable_as_input();
        match crate::arch::Family::of(arch) {
            crate::arch::Family::Cdna => Self {
                compute_units: 304,
                big: &[(usize::MAX, (32, 32))],
                small,
                unroll: false,
                shared_max: 64 << 10,
                att_band,
            },
            crate::arch::Family::Rdna => Self {
                compute_units: 40,
                big: &[(64, (Q_BLK, KV_BLK)), (128, (Q_BLK, Q_BLK))],
                small,
                unroll: false,
                shared_max: 64 << 10,
                att_band,
            },
            crate::arch::Family::Cuda => Self {
                compute_units: 28,
                big: &[(64, (Q_BLK, 2 * KV_BLK)), (128, (Q_BLK, Q_BLK))],
                small,
                unroll: true,
                shared_max: 48 << 10,
                att_band,
            },
            // Apple's threadgroup budget is half AMD's (32 KiB), and the
            // `simdgroup_matrix` accumulator feeds an operand directly, so
            // `att_band` is false and the band costs nothing. The core count is not
            // reported by Metal; `for_device` leaves this default in place.
            crate::arch::Family::Metal => Self {
                compute_units: 40,
                big: &[(usize::MAX, (Q_BLK, KV_BLK))],
                small,
                unroll: false,
                shared_max: 32 << 10,
                att_band,
            },
        }
    }

    /// [`Self::for_arch`] with the compute-unit count of the device behind
    /// `spec`, when the backend reports it.
    pub fn for_device(spec: &svod_dtype::DeviceSpec, arch: svod_dtype::GpuArch) -> Self {
        let mut policy = Self::for_arch(arch);
        if let Some(compute_units) = crate::target::compute_units(spec) {
            policy.compute_units = compute_units;
        }
        policy
    }

    /// Shared memory [`build_fa_mw_rdb`] takes for a `(q_blk, kv_blk)` tile at
    /// head dim `d`: the K and V double buffers, plus the per-warp softmax band
    /// where the arch stages it.
    pub fn shared_bytes(&self, (q_blk, kv_blk): (usize, usize), d: usize) -> usize {
        2 * 2 * kv_blk * d * IN_BYTES + if self.att_band { NUM_WARPS * kv_blk * q_blk * IN_BYTES } else { 0 }
    }

    /// The `(q_blk, kv_blk)` for a `[b, n, h, d]` attention, or `None` when even
    /// the `small` tile's buffers exceed [`Self::shared_max`].
    pub fn tile(&self, b: usize, n: usize, h: usize, d: usize) -> Option<(usize, usize)> {
        let fits = |tile| self.shared_bytes(tile, d) <= self.shared_max;
        let covers = |(q_blk, _): (usize, usize)| {
            let big_n = q_blk * NUM_WARPS;
            n.is_multiple_of(big_n) && b * h * (n / big_n) >= self.compute_units
        };
        self.big
            .iter()
            .find(|(max_d, _)| d <= *max_d)
            .map(|(_, tile)| *tile)
            .filter(|&tile| covers(tile) && fits(tile))
            .or_else(|| fits(self.small).then_some(self.small))
    }

    /// The builder config for a `[b, n, h, d]` attention; `None` as [`Self::tile`].
    pub fn config(
        &self,
        b: usize,
        n: usize,
        h: usize,
        d: usize,
        causal: bool,
        window: Option<(usize, usize)>,
    ) -> Option<FaConfig> {
        let (q_blk, kv_blk) = self.tile(b, n, h, d)?;
        Some(FaConfig { q_blk, kv_blk, unroll: self.unroll, causal, window })
    }

    /// The config for a `[b, n, h, d]` attention over `h_kv` key heads as measured
    /// on this device ([`crate::tune`]): every [`FA_TILES`] entry whose buffers fit
    /// and whose block divides `n` is timed once on synthetic operands — with the
    /// key mask when `masked` — and the fastest kept in `store`; the static
    /// [`Self::config`] choice where only one fits or nothing measured. The body
    /// form is the policy's. The launch entry consults [`crate::tune::enabled`]
    /// before coming here.
    #[allow(clippy::too_many_arguments)]
    pub fn tuned(
        &self,
        store: &crate::tune::TuneStore,
        spec: &svod_dtype::DeviceSpec,
        arch: svod_dtype::GpuArch,
        dtype: &DType,
        (b, n, h, h_kv, d): (usize, usize, usize, usize, usize),
        causal: bool,
        window: Option<(usize, usize)>,
        mask: FaMask,
        operands: FaOperands,
    ) -> Option<FaConfig> {
        let fits = |&(q_blk, kv_blk): &(usize, usize)| {
            self.shared_bytes((q_blk, kv_blk), d) <= self.shared_max && n.is_multiple_of(q_blk * NUM_WARPS)
        };
        let candidates: Vec<FaConfig> = FA_TILES
            .into_iter()
            .filter(fits)
            .map(|(q_blk, kv_blk)| FaConfig { q_blk, kv_blk, unroll: self.unroll, causal, window })
            .collect();
        let fallback = || self.config(b, n, h, d, causal, window);
        if candidates.len() < 2 {
            return fallback();
        }
        let caps = crate::ArchCaps::for_arch(arch);
        let block = (NUM_WARPS * caps.wave_size) as i64;
        let grid = |cfg: &FaConfig| [h as i64, (n / cfg.q_blk / NUM_WARPS) as i64, b as i64];
        let build = move |ker: &Kernel, cfg: FaConfig| {
            build_fa(ker, (b, n, h, h_kv, d), cfg, dtype.clone(), mask, operands);
            ker.finish(1)
        };
        let placeholders = || {
            let mut bufs: Vec<Arc<UOp>> = std::iter::once(h)
                .chain(operands.heads(h, h_kv))
                .map(|heads| UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, b * n * heads * d, dtype.clone()))
                .collect();
            if mask.key_lens {
                bufs.push(UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, b, DType::Int32));
            }
            if mask.seg_start {
                bufs.push(UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, b * n, DType::Int32));
            }
            if mask.key_mask {
                bufs.push(UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, b * n, key_mask_dtype(&caps)));
            }
            bufs
        };
        // The store line covers the candidate kernels' graphs, fingerprinted in
        // table order; only a memo miss pays for building them.
        let builds = || {
            candidates
                .iter()
                .map(|&cfg| {
                    let ker = Kernel::new(operands.kernel(), grid(&cfg), block, placeholders(), caps);
                    crate::kernel_fingerprint(&build(&ker, cfg)).digest
                })
                .collect()
        };
        let shape = [b, n, h, h_kv, d, usize::from(causal), mask.code(), dtype.bytes()];
        let key = crate::tune::TuneKey::new(operands.kernel(), spec, arch, &shape, &(&candidates, dtype));
        // One operand set for every candidate: each then reads what the previous one
        // left in the cache, as the model's attention reads the Q/K/V the kernel before
        // it just wrote. Fresh operands per candidate (25 MB each at 8×512 on gfx1201)
        // outgrow the 64 MB MALL and time every dispatch cold, which ranked `{16,16}`
        // (126 µs) over `{32,32}` (108 µs warm). Built at the first compile, i.e. on a
        // store miss only: `randn` advances the global RNG counter.
        let operands = || {
            let operand = |shape: &[usize]| Tensor::randn(shape).ok().map(|t| t.cast(dtype.clone()).to(spec.clone()));
            let mut ins = Vec::new();
            for heads in operands.heads(h, h_kv) {
                ins.push(operand(&[b, n, heads, d])?);
            }
            if mask.key_lens {
                ins.push(Tensor::full(&[b], ConstValue::Int(n as i64), DType::Int32).to(spec.clone()));
            }
            if mask.seg_start {
                ins.push(Tensor::full(&[b, n], ConstValue::Int(0), DType::Int32).to(spec.clone()));
            }
            if mask.key_mask {
                let valid = Tensor::full(&[b, n], ConstValue::Int(1), DType::Int32);
                ins.push(key_mask_operand(&valid, &caps).to(spec.clone()));
            }
            Some((ins, Tensor::empty(&[b, n, h, d], dtype.clone()).to(spec.clone())))
        };
        let mut shared = None;
        let compile = |i: usize| {
            let cfg = candidates[i];
            if shared.is_none() {
                shared = operands();
            }
            let (ins, o) = shared.as_mut()?;
            let ins: Vec<&Tensor> = ins.iter().collect();
            crate::launch::compile_kernel("flash_attention_tune", grid(&cfg), block, &mut [o], &ins, move |ker| {
                build(ker, cfg)
            })
            .ok()
        };
        store.select(&key, candidates.len(), builds, compile).map(|i| candidates[i]).or_else(fallback)
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
    let caps = crate::ArchCaps::GFX942;
    let cfg = FaPolicy::for_device(&q.device(), caps.arch)
        .config(b, n, h, d, true, None)
        .expect("the gfx942 tiles fit its LDS at every head dim the builder accepts");
    let grid = [h as i64, (n / cfg.q_blk / NUM_WARPS) as i64, b as i64];

    let in_dtype = q.uop().dtype();
    crate::run_kernel("fa_mw_rdb", grid, (NUM_WARPS * caps.wave_size) as i64, &mut [o], &[q, k, v], |ker| {
        build_fa_mw_rdb(ker, b, n, h, h_kv, d, cfg, in_dtype.clone(), FaMask::NONE);
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
/// fallback agree on every row (valid and padded alike). `key_mask` is the general
/// form of `key_lens` — any per-key pattern — and `window` restricts each query to
/// a band of keys and skips the KV blocks outside it.
#[derive(Clone, Copy)]
pub struct FaOpts<'a> {
    /// Causal (triangular) attention when `true`; full bidirectional when `false`.
    pub causal: bool,
    /// Optional `[B]` `i32` per-batch valid-key-count padding mask (key-only).
    pub key_lens: Option<&'a Tensor>,
    /// Optional `[B, N]` `i32` segment starts for packed sequences: query row
    /// `q` of batch `b` attends no key before `seg_start[b, q]`, so several
    /// sequences packed end to end in one row stay mutually invisible. Every
    /// entry must lie in `0..=q` and leave the row at least one key `key_lens`
    /// does not hide (a padding token pointing at itself stays finite); a row
    /// with no visible key is `NaN`.
    pub seg_start: Option<&'a Tensor>,
    /// Optional `[B, N]` key validity (any integer or bool dtype, non-zero =
    /// valid): key `k` of batch `b` is hidden from every query where
    /// `key_mask[b, k] == 0`, the polarity of SDPA's `key_padding_mask`. A row
    /// left with no visible key is exact zeros, as SDPA's.
    pub key_mask: Option<&'a Tensor>,
    /// Optional sliding window `(left, right)`: query `q` sees keys
    /// `q - left ..= q + right` only, and the sweep skips every KV block outside
    /// that band — the local layers of ModernBERT-style encoders. A row the band
    /// and the masks leave with no visible key is exact zeros.
    pub window: Option<(usize, usize)>,
}

impl Default for FaOpts<'_> {
    fn default() -> Self {
        Self { causal: true, key_lens: None, seg_start: None, key_mask: None, window: None }
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
///   [`crate::graph_launch`], honoring `opts.causal`, `opts.window` and the
///   optional masks (trailing globals after `o,q,k,v`: `key_lens`, `seg_start`,
///   `key_mask`).
/// - `Ok(None)` — *doesn't apply here:* the device isn't a supported arch
///   ([`FA_SUPPORTED_ARCHS`] — gfx942/gfx1151/CUDA sm_80+ with its LLVM backend), **or** the
///   runtime sequence length doesn't tile (`N % (q_blk·NUM_WARPS) != 0`). The caller
///   substitutes its own attention (e.g. [`Tensor::scaled_dot_product_attention`]).
///   The per-warp tile is the one measured fastest on this device for the shape
///   ([`FaPolicy::tuned`]; `SVOD_TK_TUNE=0` keeps the policy's static choice).
/// - `Err` — *malformed request* on a supported device: a FIXED property is wrong —
///   `q`/`k` not a statically-shaped rank-4 tensor (a dim a JIT variable pins to
///   one value counts as static), operand dtype ∉ {bf16, f16}, `D % 16 != 0`,
///   `H % H_KV != 0` (GQA), or a `key_mask` not shaped `[B, N]`. These are
///   caller bugs, raised loudly instead of silently routed to the slow path. (A
///   genuine kernel build/dispatch failure also returns `Err`.)
///
/// ```no_run
/// use svod_tensor::Tensor;
/// use svod_dtype::DType;
/// use svod_tk::FaOpts;
/// let q = Tensor::randn(&[1, 128, 16, 64]).unwrap().cast(DType::BFloat16);
/// let (k, v) = (q.clone(), q.clone());
/// // `None` ⇒ the kernel doesn't apply here; the caller picks the fallback.
/// let opts = FaOpts { causal: false, ..Default::default() };
/// if let Some(mut o) = svod_tk::flash_attention_with(&q, &k, &v, opts).unwrap() {
///     o.prepare().unwrap();
/// }
/// ```
pub fn flash_attention_with(q: &Tensor, k: &Tensor, v: &Tensor, opts: FaOpts) -> crate::LaunchResult<Option<Tensor>> {
    flash_attention_tuned(q, k, v, opts, FaPolicy::for_device)
}

/// [`flash_attention_with`] with the per-arch tile policy supplied by the caller
/// instead of read off the device ([`FaPolicy::for_device`]) — the tile-sweep
/// entry point (the analog of [`gemm_nt_with`](crate::gemm_nt_with)). `policy` is
/// consulted once per launch, its config shared by the tiling predicate and the
/// build.
pub fn flash_attention_tuned(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    opts: FaOpts,
    policy: impl Fn(&svod_dtype::DeviceSpec, svod_dtype::GpuArch) -> FaPolicy + Copy,
) -> crate::LaunchResult<Option<Tensor>> {
    let qd = crate::launch::concrete_dims(q, "flash-attention", "q", 4)?;
    let kd = crate::launch::concrete_dims(k, "flash-attention", "k", 4)?;
    let vd = crate::launch::concrete_dims(v, "flash-attention", "v", 4)?;
    let (b, n, h, d) = (qd[0], qd[1], qd[2], qd[3]);
    let h_kv = kd[2];
    let statically = crate::launch::statically;
    let (q, k, v) = (&statically(q, &qd)?, &statically(k, &kd)?, &statically(v, &vd)?);
    let dtype = q.uop().dtype();
    // The builder binds k and v to q's dtype and to `[B, N, H_kv, D]`; a mismatch
    // would pass `Kernel::gl` (which checks only the byte width) and then
    // silently change which K/V stream the body takes.
    let kv_dtype = [k, v].into_iter().map(|t| t.uop().dtype()).find(|dt| *dt != dtype);
    // K/V must agree with q on batch, KV-head count and head dim — a mismatch there
    // is a caller bug. Their SEQUENCE length is checked separately, in the tiling
    // predicate: a KV length differing from q's is cross-attention (or incremental
    // decode), a legitimate attention shape this kernel simply does not implement, so
    // it declines and the caller falls back instead of surfacing an error.
    let kv_shape = [("k", &kd), ("v", &vd)]
        .into_iter()
        .find(|(_, dims)| [dims[0], dims[2], dims[3]] != [b, h_kv, d])
        .map(|(operand, dims)| (operand, dims.clone(), vec![b, dims[1], h_kv, d]));
    let kv_seq_match = kd[1] == n && vd[1] == n;
    let layout = move || -> crate::LaunchResult<()> {
        if let Some(got) = kv_dtype {
            return crate::launch::DtypeSnafu { kernel: "flash-attention", got, expected: "the dtype of q" }.fail();
        }
        if let Some((operand, got, expected)) = kv_shape {
            return crate::launch::OperandShapeSnafu { kernel: "flash-attention", operand, expected, got }.fail();
        }
        Ok(())
    };
    launch_fa(&[q, k, v], FaOperands::Split, (b, n, h, h_kv, d), opts, policy, layout, kv_seq_match)
}

/// **Graph-native** flash attention over a fused QKV projection's output:
/// `qkv` is `[B, N, H + 2·H_kv, D]` — Q's `h` heads, then K's and V's `h_kv`
/// each — read in place ([`FaOperands::Packed`]). Split into three head views
/// instead, each would be copied into a buffer of its own before the kernel
/// runs. Options and outcomes are [`flash_attention_with`]'s, plus an `Err` when
/// `qkv`'s head count is not `h + 2·h_kv`.
pub fn flash_attention_packed(
    qkv: &Tensor,
    heads: (usize, usize),
    opts: FaOpts,
) -> crate::LaunchResult<Option<Tensor>> {
    flash_attention_packed_tuned(qkv, heads, opts, FaPolicy::for_device)
}

/// [`flash_attention_packed`] with the caller's tile policy, as
/// [`flash_attention_tuned`] is to [`flash_attention_with`].
pub fn flash_attention_packed_tuned(
    qkv: &Tensor,
    (h, h_kv): (usize, usize),
    opts: FaOpts,
    policy: impl Fn(&svod_dtype::DeviceSpec, svod_dtype::GpuArch) -> FaPolicy + Copy,
) -> crate::LaunchResult<Option<Tensor>> {
    let dims = crate::launch::concrete_dims(qkv, "flash-attention", "qkv", 4)?;
    let (b, n, d) = (dims[0], dims[1], dims[3]);
    let qkv = &crate::launch::statically(qkv, &dims)?;
    let expected = vec![b, n, h + 2 * h_kv, d];
    let layout = move || -> crate::LaunchResult<()> {
        let (operand, got) = ("qkv", dims);
        ensure!(
            got == expected,
            crate::launch::OperandShapeSnafu { kernel: "flash-attention", operand, expected, got }
        );
        Ok(())
    };
    launch_fa(&[qkv], FaOperands::Packed, (b, n, h, h_kv, d), opts, policy, layout, true)
}

/// The launch both layouts share: `ins` are the Q/K/V globals `operands`
/// binds, `layout` validates them, and `kv_seq_match` says the keys are the
/// queries' sequence (this kernel is self-attention only).
#[allow(clippy::too_many_arguments)]
fn launch_fa(
    ins: &[&Tensor],
    operands: FaOperands,
    (b, n, h, h_kv, d): (usize, usize, usize, usize, usize),
    opts: FaOpts,
    policy: impl Fn(&svod_dtype::DeviceSpec, svod_dtype::GpuArch) -> FaPolicy + Copy,
    layout: impl FnOnce() -> crate::LaunchResult<()>,
    kv_seq_match: bool,
) -> crate::LaunchResult<Option<Tensor>> {
    let statically = crate::launch::statically;
    let q = ins[0];
    let key_mask_shape = opts
        .key_mask
        .map(|m| crate::launch::concrete_dims(m, "flash-attention", "key_mask", 2))
        .transpose()?
        .filter(|dims| *dims != [b, n]);
    let dtype = q.uop().dtype();
    let dtype_ok = dtype == DType::BFloat16 || dtype == DType::Float16;
    let err_dtype = dtype.clone();
    let (tiling_device, build_device) = (q.device(), q.device());
    let tiling_dtype = dtype.clone();
    let mask = FaMask {
        key_lens: opts.key_lens.is_some(),
        seg_start: opts.seg_start.is_some(),
        key_mask: opts.key_mask.is_some(),
    };
    // The policy's config, measured on first use where tuning is on. Measuring
    // is what the first call costs, so it runs once per launch: the tiling
    // predicate and the build share its answer.
    let chosen = move |policy: &FaPolicy, device: &svod_dtype::DeviceSpec, arch, dtype: &DType| {
        if !crate::tune::enabled() {
            return policy.config(b, n, h, d, opts.causal, opts.window);
        }
        let store = crate::tune::TuneStore::global();
        policy.tuned(store, device, arch, dtype, (b, n, h, h_kv, d), opts.causal, opts.window, mask, operands)
    };
    let cfg_cell: Rc<OnceCell<Option<FaConfig>>> = Rc::default();
    let fit_cell = cfg_cell.clone();

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
            layout()?;
            if let Some(got) = key_mask_shape {
                let (operand, expected) = ("key_mask", vec![b, n]);
                return crate::launch::OperandShapeSnafu { kernel: "flash-attention", operand, expected, got }.fail();
            }
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
                h_kv > 0 && h % h_kv == 0,
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
        // legitimately not tile, so the caller falls back per-clip instead of
        // padding; a head dim whose tiles overflow shared memory declines too, as
        // does a KV length that differs from q's (this kernel is self-attention only).
        move |arch| {
            kv_seq_match
                && fit_cell
                    .get_or_init(|| chosen(&policy(&tiling_device, arch), &tiling_device, arch, &tiling_dtype))
                    .is_some_and(|cfg| n.is_multiple_of(cfg.q_blk * NUM_WARPS))
        },
        // Build for the resolved arch — caps track the real wave width.
        move |arch| {
            let caps = crate::ArchCaps::for_arch(arch);
            let cfg = cfg_cell
                .get_or_init(|| chosen(&policy(&build_device, arch), &build_device, arch, &dtype))
                .expect("checked by the tiling predicate");
            let grid = [h as i64, (n / cfg.q_blk / NUM_WARPS) as i64, b as i64];
            let out = Tensor::empty(&[b, n, h, d], dtype.clone());
            let build_dtype = dtype.clone();
            // ABI/global order is o, the Q/K/V globals, (lens), (seg_start),
            // (key_mask) — `out` is global[0], inputs map to global[1..] in order,
            // so the masks go last, in `FaMask` order.
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
            // built outside the caller's origin scope — as is the key mask's cast.
            let key_lens_clamped = opts
                .key_lens
                .map(|lens| -> crate::LaunchResult<Tensor> {
                    let _shared = svod_ir::origin::OriginScope::suspend();
                    let ones = Tensor::full(&[b], ConstValue::Int(1), DType::Int32);
                    Ok(statically(lens, &[b])?.maximum(&ones).expect("clamp key_lens >= 1"))
                })
                .transpose()?;
            let seg_start = opts.seg_start.map(|t| statically(t, &[b, n])).transpose()?;
            let key_mask = opts
                .key_mask
                .map(|m| -> crate::LaunchResult<Tensor> {
                    let _shared = svod_ir::origin::OriginScope::suspend();
                    Ok(key_mask_operand(&statically(m, &[b, n])?, &caps))
                })
                .transpose()?;
            let mut ins: Vec<&Tensor> = ins.to_vec();
            ins.extend([&key_lens_clamped, &seg_start, &key_mask].into_iter().flatten());
            let block = (NUM_WARPS * caps.wave_size) as i64;
            crate::graph_launch(operands.kernel(), grid, block, out, &ins, caps, move |ker| {
                build_fa(ker, (b, n, h, h_kv, d), cfg, build_dtype.clone(), mask, operands);
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
/// let q = Tensor::randn(&[1, 128, 16, 64]).unwrap().cast(DType::BFloat16);
/// let (k, v) = (q.clone(), q.clone());
/// if let Some(mut o) = svod_tk::flash_attention(&q, &k, &v).unwrap() {
///     o.prepare().unwrap();
/// }
/// ```
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> crate::LaunchResult<Option<Tensor>> {
    flash_attention_with(q, k, v, FaOpts::default())
}
