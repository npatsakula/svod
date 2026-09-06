//! Fused brute-force k-means **assignment** ([`build_kmeans_assign`]) and its
//! public lazy-[`Tensor`] entry ([`kmeans_assign`]), plus the generic-graph
//! centroid-update entry ([`kmeans_update`]).
//!
//! **Assignment** mirrors the x²-free KNN score ([`crate::kernels::knn`]):
//! `score[k, n] = ‖c[k]‖² − 2·⟨x[n], c[k]⟩`. The per-point `‖x[n]‖²` self-term
//! is dropped (constant under the argmin over centroids `k`). The dominant
//! `‖c[k]‖²` (`c_sq`) is precomputed in **f32** outside the kernel and threaded
//! in bf16-augmentation-free (same trick as KNN — an f32 `c_sq` smuggled through
//! a bf16 WMMA operand would lose its precision).
//!
//! The assignment kernel streams `K` centroids in [`TM`]-tall tiles through a
//! [`crate::loop_scope::Loop`] and keeps, per point, the running argmin — a
//! flashlib-style top-1 fold (the degenerate case of KNN's argmin-insert top-K).
//! Each tile produces a `(tile_min, tile_arg)` pair via [`Group::row_arg_reduce`];
//! the running best is extracted from the carried `[BLK, BLK]` val tile (also via
//! `row_arg_reduce(Min)` — padding rows at `+∞` never win), and the per-point
//! predicate `tile_min < running_best` gates a single-slot `map_position` update
//! (the KNN `evict_slot` pattern with a fixed slot-0 evict target).
//!
//! **Update** ([`kmeans_update`]) is the generic-graph peer: `scatter_reduce(Sum)`
//! for per-cluster sums/counts, divide, empty-cluster fixup, and per-cluster
//! shift — no tile kernel (sort/scatter don't tile well; the graph is the right
//! level). This mirrors the tk philosophy: fuse what's worth fusing (the O(N·K·D)
//! assign), leave the rest to the scheduler.
//!
//! Orientation (same as KNN): **centroids `k` are the reduced / row axis** and
//! points `n` the column, so `row_arg_reduce(Min)` folds the centroid axis and
//! returns one `(val, idx)` per point. The cross MMA is `mma_atb(score, cᵀ, xᵀ)`
//! — both operands transposed to `[d, *]` Col fragments — yielding
//! `score[k, n] = Σ_d c[k,d]·x[n,d] = ⟨c[k], x[n]⟩` in the f32 accumulator.

use std::sync::Arc;

use snafu::{ResultExt, ensure};
use svod_dtype::DType;
use svod_ir::{ConstValue, UOp};
use svod_tensor::Tensor;
use svod_tensor::indexing::ScatterReduction;

use crate::ArgDir;
use crate::Group;
use crate::arch::FragRole;
use crate::group::MoveIdx;
use crate::index::{Idx, cidx, load_at};
use crate::kernel::Kernel;
use crate::scaffold::GlSpec;
use crate::tile::{GL, RT, RV, RegTile};
use crate::tiles::{TileLayout, VecLayout};

/// The WMMA tile edge (K=16); the cross MMA operates on 16×16 fragments, so the
/// point / centroid / D dims must each be a multiple of it. Also the per-workgroup
/// point-block width (each wave owns [`BLK`] points, selected by `block_idx[0]`).
const BLK: usize = 16;

/// The centroid stream-tile height — a multiple of [`BLK`] (16). Centroids are
/// streamed in `K/TM` tiles through the assignment loop. `16` (= BLK) keeps the
/// centroid operand fragments small: profiling showed the taller `32`/`64` tiles
/// push VGPR past the occupancy cliff (248/256 VGPR → 38% occ), whereas `16`
/// holds ~176 VGPR → 50% occ and runs ~1.7× faster despite the extra passes.
const TM: usize = 16;

/// The GPU arch(es) this kernel targets (gfx942 CDNA3 wave64 + gfx1151 RDNA3.5
/// wave32). The launcher gates against this list.
/// Validated on gfx942 (CDNA3) and gfx1151 (RDNA3.5).
pub const KMEANS_SUPPORTED_ARCHS: crate::ArchSet =
    crate::ArchSet::amd(&[svod_dtype::AmdArch::Gfx942, svod_dtype::AmdArch::Gfx1151]);

const POS_INF: f64 = f64::INFINITY;

fn fconst(dt: &DType, v: f64) -> Arc<UOp> {
    UOp::const_(dt.clone(), ConstValue::Float(v))
}
fn iconst32(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(v))
}

/// Anchor a register tile's next op on the centroid-loop range carried by
/// `k_blk` (cf. KNN's `reinit_on`): `t.after([k_blk])` when `k_blk` is the
/// rolled loop index, a no-op for a `Const` tile index (single-tile path).
fn reinit_on<'k>(t: RT<'k>, k_blk: &Idx) -> RT<'k> {
    match k_blk {
        Idx::Uop(u) => t.after(u),
        Idx::Const(_) => t,
    }
}

/// Round `x` up to the next multiple of [`BLK`] (16) — the WMMA tile edge.
fn pad16(x: usize) -> usize {
    x.div_ceil(BLK) * BLK
}

/// Read a per-point RV scalar inside a `map_position` over a Col `[*, BLK]` tile:
/// the point is the tile's column / width axis, so it selects RV slot `idx[1]`
/// (cf. KNN's `rv_query_src`). The RV buffer is anchored so a constant-address
/// read of a carried (loop) RV is not hoisted out of the centroid loop.
fn rv_point_src<'k>(warp: &Group<'k>, rv: &RV<'k>) -> (Arc<UOp>, Vec<usize>) {
    (warp.anchor(rv.uop()), rv.shape().to_vec())
}

/// A length-`BLK` f32 RV seeded to `init` (the `row_arg_reduce` value accumulator).
fn seed_val<'k>(ker: &'k Kernel, warp: &Group<'k>, init: f64) -> RV<'k> {
    let frag = ker.frag(FragRole::Accumulator);
    warp.clear_rv(ker.rv(BLK, DType::Float32, VecLayout::Ortho, frag), init)
}
/// A length-`BLK` Int32 index RV seeded to `−1` (the `row_arg_reduce` index acc).
fn seed_idx<'k>(ker: &'k Kernel, warp: &Group<'k>) -> RV<'k> {
    let frag = ker.frag(FragRole::Accumulator);
    warp.clear_rv(ker.rv(BLK, DType::Int32, VecLayout::Ortho, frag), -1.0)
}

// =============================================================================
// Stage 1 — the score tile.
// =============================================================================

/// The cross-term + `c_sq` combine yielding one `[TM, BLK]` Col f32 score tile
/// `score[k, n] = ‖c[k]‖² − 2·⟨x[n], c[k]⟩` for centroid rows
/// `[k_blk·TM, +TM)` — the Stage-1 machinery, factored so the loop body calls it
/// per [`TM`]-tall centroid tile. `x_reg_t` is the loop-invariant point operand
/// `[d, BLK]` (the caller loads it once); `k_blk` is the centroid tile index.
/// `masked` gates the GLOBAL→LDS/REG hops against the true centroid extent so a
/// ragged final tile reads `0.0` instead of touching out-of-bounds memory (the
/// caller then masks those rows to `+∞` for the argmin).
#[allow(clippy::too_many_arguments)]
fn score_tile<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    query: usize,
    d: usize,
    x_reg_t: &RT<'k>,
    c_gl: &GL,
    c_sq_gl: &GL,
    k_blk: &Idx,
    masked: bool,
) -> RT<'k> {
    let bf16 = DType::BFloat16;
    let (row, col) = (TileLayout::Row, TileLayout::Col);

    // GLOBAL(centroid tile k_blk) → LDS (swizzled) → REG, transposed to the
    // `[d, TM]` Col operand for the contraction over D (mirrors KNN's cᵀ).
    let c_smem = ker.shared_sw((TM, d), bf16.clone(), row);
    let c_reg = ker.operand((TM, d), bf16.clone(), row);
    let c_reg_t = ker.operand((d, TM), bf16.clone(), col);

    let c_smem = warp.load(c_smem, c_gl.clone(), MoveIdx::block((0, 0, k_blk.clone(), 0), 2));
    let c_reg = warp.load(c_reg, c_smem, MoveIdx::default());
    let c_reg_t = warp.transpose(c_reg_t, &c_reg);

    // score[k, n] = Σ_d c[k,d]·x[n,d] = ⟨c[k], x[n]⟩ (centroid = row, point = col).
    // Re-zero the accumulator each centroid iteration (cf. KNN's score_tile).
    let cross = warp.zero(reinit_on(ker.acc((TM, query), col), k_blk));
    let cross = warp.mma_atb(cross, &c_reg_t, x_reg_t);

    // Load c_sq[k_blk] into the SAME accumulator fragment + layout as the cross
    // MMA output, so the two align lane-for-lane.
    let cs_mi = MoveIdx::block((0, 0, k_blk.clone(), 0), 2);
    let cs_mi = if masked { cs_mi.masked() } else { cs_mi };
    let c_sq = warp.load(ker.acc((TM, query), col), c_sq_gl.clone(), cs_mi);

    // score = c_sq − 2·cross, all f32.
    let cross = warp.mul_scalar(cross, -2.0);
    warp.add(cross, &c_sq)
}

/// Load the point tile `[BLK, d]` and transpose it to its `[d, BLK]` Col operand
/// fragment for the cross contraction over D — loop-invariant, so the builder
/// loads it once. `n_blk` is the point-block index this workgroup owns (in
/// BLK-unit offsets): `block_idx[0]` selects each 16-point block of a wide
/// `[Npad, d]` point input.
fn load_points_t<'k>(ker: &'k Kernel, warp: &Group<'k>, d: usize, x_gl: &GL, n_blk: &Idx) -> RT<'k> {
    let bf16 = DType::BFloat16;
    let (row, col) = (TileLayout::Row, TileLayout::Col);
    let x_smem = ker.shared_sw((BLK, d), bf16.clone(), row);
    let x_reg = ker.operand((BLK, d), bf16.clone(), row);
    let x_reg_t = ker.operand((d, BLK), bf16.clone(), col);
    let x_smem = warp.load(x_smem, x_gl.clone(), MoveIdx::block((0, 0, n_blk.clone(), 0), 2));
    let x_reg = warp.load(x_reg, x_smem, MoveIdx::default());
    warp.transpose(x_reg_t, &x_reg)
}

// =============================================================================
// Stage 2 — the running argmin fold.
// =============================================================================

/// Per-point running argmin state: two Col-layout `[BLK, BLK]` register tiles —
/// `val` (f32, row 0 = running best, rows 1-15 = `+∞` padding) and `idx` (Int32,
/// row 0 = running arg, rows 1-15 = `−1`). The padding rows stay at `+∞`/`−1` so
/// `row_arg_reduce(Min)` extracting the running best always picks row 0.
struct Best<'k> {
    val: RT<'k>,
    idx: RT<'k>,
}

/// Mask ragged centroid rows (`global_k ≥ k_clusters`) of a Col `[TM, BLK]`
/// score tile to `+∞` via [`Group::mask_where`], so the per-point argmin never
/// selects the padding the masked score load zeroed (cf. KNN's
/// `mask_ragged_rows`).
fn mask_ragged_centroids<'k>(warp: &Group<'k>, score: RT<'k>, k_tile: &Arc<UOp>, k_clusters: usize) -> RT<'k> {
    let bound = cidx(k_clusters as i64);
    warp.mask_where(score, Idx::Uop(k_tile.clone()), Idx::Const(0), POS_INF, move |global_k, _| global_k.ge(&bound))
}

/// Update slot 0 of a `[BLK, BLK]` Col tile at each point-column where
/// `candidate[n] < current_best[n]`, writing `replacement[n]`. The per-element
/// `k_pos` (= row position) gates the update to row 0 only — rows 1-15 keep their
/// `+∞`/`−1` padding, so the next iteration's `row_arg_reduce(Min)` still picks
/// row 0. The per-point RVs are read by the column index (`idx[1]`), the
/// multi-RV generalization of KNN's `combine_rv`.
fn update_slot0<'k>(
    warp: &Group<'k>,
    tile: RT<'k>,
    candidate: &RV<'k>,
    current_best: &RV<'k>,
    replacement: &RV<'k>,
) -> RT<'k> {
    let (cand_buf, cand_shape) = rv_point_src(warp, candidate);
    let (best_buf, best_shape) = rv_point_src(warp, current_best);
    let (repl_buf, repl_shape) = rv_point_src(warp, replacement);
    warp.map_position(tile, Idx::Const(0), Idx::Const(0), move |x, idx, k_pos, _col| {
        let k_pos = k_pos.cast(DType::Int32);
        let n = idx[1].clone();
        let cand = load_at(&cand_buf, &cand_shape, &[n.clone(), Idx::Const(0)]);
        let best = load_at(&best_buf, &best_shape, &[n.clone(), Idx::Const(0)]);
        let mut rpl = load_at(&repl_buf, &repl_shape, &[n, Idx::Const(0)]);
        if rpl.dtype() != x.dtype() {
            rpl = rpl.cast(x.dtype());
        }
        let do_update = cand.lt(&best);
        let is_slot0 = k_pos.eq(&iconst32(0));
        let hit = is_slot0.and_(&do_update);
        UOp::try_where(hit, rpl, x.clone()).expect("update_slot0 where")
    })
}

/// One centroid tile's argmin fold: compute the score sub-tile, mask ragged
/// centroids, find the per-point tile-min, and conditionally update the running
/// best's slot 0 (cf. KNN's `topk_insert`, degenerated to top-1).
#[allow(clippy::too_many_arguments)]
fn fold_best<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    k_clusters: usize,
    d: usize,
    x_reg_t: &RT<'k>,
    c_gl: &GL,
    c_sq_gl: &GL,
    k_tile: &Arc<UOp>,
    masked: bool,
    mut best: Best<'k>,
) -> Best<'k> {
    // score[k, n] for this centroid tile — `TM`-tall (`TM/16` stacked frags),
    // centroid = row, point = col.
    let mut score = score_tile(ker, warp, BLK, d, x_reg_t, c_gl, c_sq_gl, &Idx::Uop(k_tile.clone()), masked);
    if masked {
        score = mask_ragged_centroids(warp, score, k_tile, k_clusters);
    }

    // a. per-point tile-min over the TM centroid rows → (val, arg) RVs.
    let (tile_min, tile_arg) =
        warp.row_arg_reduce(seed_val(ker, warp, POS_INF), seed_idx(ker, warp), &score, ArgDir::Min);

    // b. global_k = k_tile·TM + tile_arg in a FRESH RV (cf. KNN's global_m).
    let kbase = k_tile.mul(&cidx(TM as i64)).cast(DType::Int32);
    let (ta_buf, ta_shape) = (warp.anchor(tile_arg.uop()), tile_arg.shape().to_vec());
    let global_k = warp
        .map(seed_idx(ker, warp), move |_, idx| load_at(&ta_buf, &ta_shape, idx).try_add(&kbase).expect("global_k"));

    // c. per-point running best (extracted from the val tile via row_arg_reduce).
    //    Padding rows at +∞ never win the min, so the fold yields row-0's value.
    let (running_best, _) =
        warp.row_arg_reduce(seed_val(ker, warp, POS_INF), seed_idx(ker, warp), &best.val, ArgDir::Min);

    // d. Conditional update at slot 0: where tile_min < running_best, write
    //    tile_min (val) and global_k (idx). Chain idx after val so both carried
    //    writes share one loop END (cf. KNN's evict_slot chaining).
    best.val = update_slot0(warp, best.val, &tile_min, &running_best, &tile_min);
    best.idx = update_slot0(warp, best.idx.after(&best.val), &tile_min, &running_best, &global_k);
    best
}

/// Store the running argmin to the `[1, 1, N_pad, 1]` outputs. The running tiles
/// are Col `[BLK, BLK]`; the output wants `[BLK_points, 1]` (the 1 live K-slot),
/// the transpose of `[BLK, BLK]` → Row `[BLK, BLK]` (AccumulatorT fragment),
/// stored with the boundary mask dropping the columns `≥ 1` (only slot 0 is
/// live). Both transposes come BEFORE both global stores, so the kernel's final
/// two terminal stores — popped by `finish(2)` — are exactly the two output
/// writes (cf. KNN's `store_topk` with `k = 1`).
fn store_best<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    ids_gl: &GL,
    val_gl: &GL,
    best_val: &RT<'k>,
    best_idx: &RT<'k>,
    n_blk: &Idx,
) {
    let row = TileLayout::Row;
    let acc_t = ker.frag(FragRole::AccumulatorT);
    // k = 1 < BLK, so the trailing columns must be masked off.
    let mi = MoveIdx::block((0, 0, n_blk.clone(), 0), 2).masked();

    let val_t = warp.transpose(ker.acc_t((BLK, BLK), row), best_val);
    let idx_t = warp.transpose(ker.rt((BLK, BLK), DType::Int32.clone(), row, acc_t), best_idx);
    let _ = warp.store(val_gl.clone(), val_t, mi.clone());
    let _ = warp.store(ids_gl.clone(), idx_t, mi);
}

/// Build the k-means assignment kernel into the bound ABI.
///
/// ABI (outputs then inputs, fixed by [`Kernel::bind_abi`]):
/// - `cluster_ids` (`[1, 1, N, 1]`, Int32) — the nearest-centroid index per point.
/// - `best_dist` (`[1, 1, N, 1]`, f32) — the x²-free score at that centroid.
/// - `x` (`[1, 1, N, d]`, bf16) — the point rows.
/// - `c` (`[1, 1, K, d]`, bf16) — the centroid rows.
/// - `c_sq_rep` (`[1, 1, K, BLK]`, f32) — `‖c[k]‖²` precomputed outside the kernel
///   and replicated along the point axis (each `(k, n)` holds `c_sq[k]`).
///
/// Single-warp; `N`, `d` must each be a multiple of [`BLK`] (16). Each workgroup
/// processes one [`BLK`]-point block selected by `block_idx[0]`; `K` is streamed
/// in [`TM`]-tall tiles (ragged-K is masked). Built **rolled** (`arg_reduce`
/// panics under unroll).
///
/// # Panics
/// Panics unless `N` and `d` are each a multiple of 16.
pub fn build_kmeans_assign(ker: &Kernel, n_points: usize, k_clusters: usize, d: usize) {
    Kernel::assert_divisible(n_points, BLK, "kmeans N");
    Kernel::assert_divisible(d, BLK, "kmeans D");
    Kernel::assert_divisible(TM, BLK, "kmeans TM");
    assert!(k_clusters > 0, "kmeans K must be > 0");

    let bf16 = DType::BFloat16;
    let f32 = DType::Float32;
    let i32 = DType::Int32;
    let col = TileLayout::Col;
    let warp = ker.warp();

    // ABI: outputs (ids i32, dist f32) then inputs (x, c — bf16; c_sq_rep — f32).
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[1, 1, n_points, 1], i32.clone()), GlSpec::new(&[1, 1, n_points, 1], f32.clone())],
        &[
            GlSpec::new(&[1, 1, n_points, d], bf16.clone()),
            GlSpec::new(&[1, 1, k_clusters, d], bf16.clone()),
            GlSpec::new(&[1, 1, k_clusters, BLK], f32.clone()),
        ],
    );
    let (ids_gl, val_gl, x_gl, c_gl, c_sq_gl): (GL, GL, GL, GL, GL) =
        (outs[0].clone(), outs[1].clone(), ins[0].clone(), ins[1].clone(), ins[2].clone());

    // Point-block grid tiling: each workgroup owns one 16-point block.
    let n_blk = Idx::Uop(ker.block_idx[0].clone());
    let x_reg_t = load_points_t(ker, &warp, d, &x_gl, &n_blk);

    // Running argmin state (Col `[BLK, BLK]`). ALL rows seed to +∞ (val) / -1
    // (idx); row 0 is updated by `fold_best`, rows 1-15 stay as padding.
    let best_val = warp.map(ker.acc((BLK, BLK), col), |x, _| fconst(&x.dtype(), POS_INF));
    let best_idx = warp.map(ker.rt((BLK, BLK), i32.clone(), col, ker.frag(FragRole::Accumulator)), |_, _| iconst32(-1));
    let best = Best { val: best_val, idx: best_idx };

    // Stream K centroids in TM-tall tiles via the FA running-state Loop carry.
    let tiles = k_clusters.div_ceil(TM);
    let masked = !k_clusters.is_multiple_of(TM);
    let lp = ker.loop_static(tiles as i64);
    let k_tile = lp.index().clone();
    let best = Best { val: lp.reinit(best.val), idx: lp.reinit(best.idx) };

    let best = fold_best(ker, &warp, k_clusters, d, &x_reg_t, &c_gl, &c_sq_gl, &k_tile, masked, best);

    // Close the loop once; both carried tiles read their post-loop value via
    // `.after([end])` (cf. KNN's `lp.close()` + `.after(&ended)`).
    let ended = lp.close();
    let val_after = best.val.after(&ended);
    let idx_after = best.idx.after(&ended);

    store_best(ker, &warp, &ids_gl, &val_gl, &val_after, &idx_after, &n_blk);
}

// =============================================================================
// Stage 3 — the public lazy-Tensor entries + the generic-graph tail.
// =============================================================================

/// **Graph-native** fused brute-force k-means assignment — the matmul/KNN peer
/// for k-means, returning lazy output [`Tensor`]s (the tile kernel is a
/// `custom_kernel` / `Op::Call` node, the `‖x‖²` re-add is an ordinary
/// generic-graph op).
///
/// For `N` points `x` (`[N, D]`) and `K` centroids `c` (`[K, D]`, **any float
/// dtype**) it returns `Some((cluster_ids, best_dist))`:
/// - `cluster_ids` (`[N]`, i32) — the index of the nearest centroid per point.
/// - `best_dist` (`[N]`, f32) — the **true** squared-L2 distance
///   `‖x[n] − c[cluster_ids[n]]‖²` (the kernel's x²-free score + the re-added
///   `‖x‖²` self-term, clamped ≥ 0).
///
/// The kernel streams the centroids and keeps the running argmin from the x²-free
/// score `‖c[k]‖² − 2·⟨x[n],c[k]⟩` ([`build_kmeans_assign`]); this entry owns the
/// host-side prep (cast → bf16, zero-pad `N`/`D` to the WMMA edge, the f32
/// `‖c‖²`) and the generic-graph tail (slice the padding, re-add `‖x‖²`).
///
/// Like [`crate::knn`] / [`crate::matmul`], the outcome is three-way:
/// - `Ok(Some((cluster_ids, best_dist)))` — ran (lazy nodes; `prepare()` to realize).
/// - `Ok(None)` — the device isn't a supported arch ([`KMEANS_SUPPORTED_ARCHS`] —
///   gfx942 / gfx1151 with the AMD toolchain). The caller substitutes its own path.
/// - `Err` — a malformed request on a supported device: `x`/`c` not statically-shaped
///   rank-2 tensors, or mismatched `D`. These are caller bugs.
///
/// ```no_run
/// use svod_tensor::Tensor;
/// let x = Tensor::randn(&[1024, 64]).unwrap(); // 1024 points, dim 64
/// let c = Tensor::randn(&[32, 64]).unwrap();   // 32 centroids
/// if let Some((mut ids, mut dists)) = svod_tk::kmeans_assign(&x, &c).unwrap() {
///     ids.prepare().unwrap();    // [1024] i32 nearest-centroid index per point
///     dists.prepare().unwrap();  // [1024] f32 squared-L2 distance to it
/// }
/// ```
pub fn kmeans_assign(x: &Tensor, c: &Tensor) -> crate::LaunchResult<Option<(Tensor, Tensor)>> {
    let xd = crate::launch::concrete_dims(x, "kmeans_assign", "x", 2)?;
    let cd = crate::launch::concrete_dims(c, "kmeans_assign", "c", 2)?;
    let (n, dx) = (xd[0], xd[1]);
    let (k, dc) = (cd[0], cd[1]);

    // Structural validity (`Err`) — checked BEFORE arch resolution.
    ensure!(dx == dc, crate::launch::OperandDimMismatchSnafu { kernel: "kmeans_assign", dim: "D", a: dx, b: dc });

    // Three-way policy (inlined — multi-output; `launch_custom` is single-Tensor).
    let Some(arch) = crate::target::resolve_supported_arch(&x.device(), KMEANS_SUPPORTED_ARCHS).ok() else {
        return Ok(None);
    };

    let caps = crate::ArchCaps::for_arch(arch);
    let (f32, bf16) = (DType::Float32, DType::BFloat16);
    let d_pad = pad16(dx);
    let n_pad = pad16(n);

    // f32 copies for the tail (‖x‖² re-add).
    let x_f32 = x.cast(f32.clone()).context(crate::launch::OperandSnafu)?;

    // Kernel bf16 operands, zero-padded to the WMMA edge. `K` is NOT padded —
    // the kernel ragged-masks its final centroid tile.
    let x_bf = pad_operand(&x.cast(bf16.clone()).context(crate::launch::OperandSnafu)?, n, dx, n_pad, d_pad)?;
    let c_bf = pad_operand(&c.cast(bf16.clone()).context(crate::launch::OperandSnafu)?, k, dc, k, d_pad)?;

    // c_sq[k] = Σ_d c[k,d]² in f32, replicated to [1,1,K,BLK].
    let c_sq_rep = c_sq_replicated(&c.cast(f32.clone()).context(crate::launch::OperandSnafu)?, k)?;

    let ids_t = Tensor::empty(&[1, 1, n_pad, 1], DType::Int32);
    let val_t = Tensor::empty(&[1, 1, n_pad, 1], f32.clone());
    let grid = [(n_pad / BLK) as i64, 1, 1];
    let block = caps.wave_size as i64;

    let outs = crate::graph_launch_multi(
        "kmeans_assign",
        grid,
        block,
        vec![ids_t, val_t],
        &[&x_bf, &c_bf, &c_sq_rep],
        caps,
        move |ker| {
            build_kmeans_assign(ker, n_pad, k, d_pad);
            ker.finish(2)
        },
    )?;
    let (ids_raw, val_raw) = (outs[0].clone(), outs[1].clone());

    kmeans_assign_tail(&ids_raw, &val_raw, &x_f32, n).map(Some)
}

/// Zero-pad a `[rows, d]` tensor's last (`D`) axis to `d_pad` and its leading
/// (row) axis to `rows_pad`, then add the kernel's `[1, 1, …]` leading singleton
/// axes — the bf16 kernel operand layout (cf. KNN's `pad_operand`).
fn pad_operand(t: &Tensor, rows: usize, d: usize, rows_pad: usize, d_pad: usize) -> crate::LaunchResult<Tensor> {
    let padded = t
        .try_pad(&[(0, (rows_pad - rows) as isize), (0, (d_pad - d) as isize)])
        .context(crate::launch::OperandSnafu)?;
    padded.try_reshape([1isize, 1, rows_pad as isize, d_pad as isize]).context(crate::launch::OperandSnafu)
}

/// `c_sq[k] = Σ_d c_f32[k,d]²` in f32, replicated to the kernel's `[1, 1, K, BLK]`
/// `c_sq_rep` operand (cf. KNN's `c_sq_replicated`).
fn c_sq_replicated(c_f32: &Tensor, k: usize) -> crate::LaunchResult<Tensor> {
    let c_sq = c_f32
        .try_mul(c_f32)
        .context(crate::launch::OperandSnafu)?
        .sum_with()
        .axes(1isize)
        .keepdim(true)
        .call()
        .context(crate::launch::OperandSnafu)?; // [K, 1]
    c_sq.try_reshape([1isize, 1, k as isize, 1])
        .context(crate::launch::OperandSnafu)?
        .try_expand([1isize, 1, k as isize, BLK as isize])
        .context(crate::launch::OperandSnafu)
}

/// The generic-graph tail: slice off the padded point rows, re-add `‖x‖²` to the
/// x²-free score for the true squared-L2 distance, and clamp ≥ 0. Returns
/// `(cluster_ids [N] i32, best_dist [N] f32)`.
fn kmeans_assign_tail(
    ids_raw: &Tensor,
    val_raw: &Tensor,
    x_f32: &Tensor,
    n: usize,
) -> crate::LaunchResult<(Tensor, Tensor)> {
    let op = crate::launch::OperandSnafu;

    // 1. [1,1,Npad,1] → [Npad,1] → [N,1] → [N] for ids; keep score as [N,1] for
    //    the broadcast-safe add with x_sq [N,1] below.
    let ids = ids_raw
        .try_reshape([-1, 1isize])
        .context(op)?
        .try_shrink([(0, n as isize), (0, 1isize)])
        .context(op)?
        .try_reshape([n as isize])
        .context(op)?;
    let score =
        val_raw.try_reshape([-1, 1isize]).context(op)?.try_shrink([(0, n as isize), (0, 1isize)]).context(op)?; // [N, 1]

    // 2. True ‖x−c‖² = ‖x‖² + (c_sq − 2·⟨x,c⟩) = ‖x‖² + score. Clamp ≥ 0 (bf16
    //    rounding in the cross term can make the raw sum slightly negative).
    let x_sq = x_f32.try_mul(x_f32).context(op)?.sum_with().axes(1isize).keepdim(true).call().context(op)?; // [N, 1]
    let dist = score.try_add(&x_sq).context(op)?.relu().context(op)?.try_reshape([n as isize]).context(op)?;

    Ok((ids, dist))
}

/// **Generic-graph** k-means centroid update — the second half of a Lloyd
/// iteration. Given the point assignments, computes new centroids by averaging
/// each cluster's members, with empty-cluster fixup (reuse the old centroid) and
/// a per-cluster shift (for convergence checks).
///
/// This is pure generic-graph (no tile kernel): `scatter_reduce(Sum)` for
/// per-cluster sums/counts, divide, fixup, and shift. The sort/scatter pattern
/// doesn't tile well — the graph is the right level. The caller owns the Lloyd
/// loop (alternate [`kmeans_assign`] and [`kmeans_update`] until convergence).
///
/// Returns `(new_centroids, shift)`:
/// - `new_centroids` (`[K, D]`, f32) — the updated centroids. Empty clusters
///   reuse their old centroid (not zeroed).
/// - `shift` (`[K]`, f32) — `‖new_centroids[k] − old_centroids[k]‖₂`, the
///   per-cluster displacement for convergence checking.
///
/// ```no_run
/// use svod_tensor::Tensor;
/// let x = Tensor::randn(&[1024, 64]).unwrap();       // 1024 points
/// let c = Tensor::randn(&[32, 64]).unwrap();          // 32 centroids
/// if let Some((ids, _dist)) = svod_tk::kmeans_assign(&x, &c).unwrap() {
///     let (new_c, shift) = svod_tk::kmeans_update(&x, &ids, &c).unwrap();
///     // shift.max() < tol ⇒ converged
/// }
/// ```
pub fn kmeans_update(
    x: &Tensor,
    cluster_ids: &Tensor,
    old_centroids: &Tensor,
) -> crate::LaunchResult<(Tensor, Tensor)> {
    let xd = crate::launch::concrete_dims(x, "kmeans_update", "x", 2)?;
    let (n, d) = (xd[0], xd[1]);
    let idd = crate::launch::concrete_dims(cluster_ids, "kmeans_update", "cluster_ids", 1)?;
    let od = crate::launch::concrete_dims(old_centroids, "kmeans_update", "old_centroids", 2)?;
    let (k, d2) = (od[0], od[1]);

    // Structural validity (`Err`).
    ensure!(d == d2, crate::launch::OperandDimMismatchSnafu { kernel: "kmeans_update", dim: "D", a: d, b: d2 });
    ensure!(n == idd[0], crate::launch::OperandDimMismatchSnafu { kernel: "kmeans_update", dim: "N", a: n, b: idd[0] });

    let op = crate::launch::OperandSnafu;
    let f32 = DType::Float32;

    let xf = x.cast(f32.clone()).context(op)?;
    let ocf = old_centroids.cast(f32.clone()).context(op)?;

    // Per-cluster counts: scatter_reduce(Sum) of ones at cluster_ids → [K].
    let ones_n = Tensor::full(&[n], ConstValue::Float(1.0), f32.clone()).context(op)?;
    let counts = Tensor::full(&[k], ConstValue::Float(0.0), f32.clone())
        .context(op)?
        .scatter_reduce(0, cluster_ids, &ones_n, ScatterReduction::Sum, false)
        .context(op)?; // [K]

    // Per-cluster sums: scatter_reduce(Sum) of x at cluster_ids → [K, D].
    let ids_n1 = cluster_ids.try_reshape([n as isize, 1isize]).context(op)?;
    let ids_expand = ids_n1.try_expand([n as isize, d as isize]).context(op)?;
    let sums = Tensor::full(&[k, d], ConstValue::Float(0.0), f32.clone())
        .context(op)?
        .scatter_reduce(0, &ids_expand, &xf, ScatterReduction::Sum, false)
        .context(op)?; // [K, D]

    // new_centroids = where(counts > 0, sums / counts.clamp_min(1), old_centroids).
    let counts_kd =
        counts.try_reshape([k as isize, 1isize]).context(op)?.try_expand([k as isize, d as isize]).context(op)?;
    let one = Tensor::from_slice([1.0f32]);
    let zero = Tensor::from_slice([0.0f32]);
    let safe_counts = counts_kd.maximum(&one).context(op)?;
    let divided = sums.try_div(&safe_counts).context(op)?;
    let has_members = counts_kd.try_gt(&zero).context(op)?;
    let new_centroids = divided.where_(&has_members, &ocf).context(op)?;

    // shift[k] = ‖new[k] − old[k]‖₂.
    let diff = new_centroids.try_sub(&ocf).context(op)?;
    let diff_sq = diff.try_mul(&diff).context(op)?;
    let shift = diff_sq.sum_with().axes(1isize).dtype(f32.clone()).call().context(op)?.try_sqrt().context(op)?; // [K]

    Ok((new_centroids, shift))
}
