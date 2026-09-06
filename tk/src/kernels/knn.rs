//! A fused brute-force KNN: the two tile-kernel stages plus the public
//! lazy-[`Tensor`] entry point [`knn`] (its host-side prep + generic-graph tail).
//!
//! **Stage 1** ([`build_knn_score`]) is the **x²-free score tile**. For query
//! rows `x[query, d]` and corpus rows `c[corpus, d]` the score is
//! `score[m, n] = ‖c[m]‖² − 2·⟨x[n], c[m]⟩`. The query self-term `‖x[n]‖²` is
//! dropped (it is constant per query row `n`, so it never changes the argmin over
//! the corpus `m` that the running top-K in Stage 2 takes). The dominant distance
//! term `‖c[m]‖²` (`c_sq`) is precomputed in **f32** outside the kernel and passed
//! in (an augmentation that smuggled it through a bf16 WMMA operand would lose its
//! precision), replicated along the query axis so every `(m, n)` reads `c_sq[m]`.
//!
//! **Stage 2** ([`build_knn_topk`]) streams the corpus in [`TM`]-tall tiles and
//! keeps, per query, the running unsorted top-K nearest corpus rows via a
//! flashlib-style **argmin-insert**: no score recompute, no in-kernel sort. The
//! final K-ordering is offloaded to the generic graph in Stage 3.
//!
//! Orientation (mirrors [`crate::kernels::fa::fa_qk`]'s `QKᵀ`): the **corpus `m`
//! is the reduced / row axis** and the query `n` the column, so Stage 2's running
//! top-K over the corpus folds the score tile's row — the inner-carrying axis on
//! both the gfx942 normal accumulator (matrix-col reduce) and the gfx1151 wave32
//! even/odd interleave accumulator (matrix-row reduce); the caller arranges the
//! tile, exactly as FA does. The cross MMA is `mma_atb(score, cᵀ, xᵀ)` (both
//! operands the corpus/query tiles transposed to `[d, *]` Col fragments), giving
//! `score[m, n] = Σ_d c[m,d]·x[n,d] = ⟨c[m], x[n]⟩` in the f32 accumulator.
//!
//! The `c_sq` global is loaded into a tile declared with the SAME accumulator
//! fragment as the cross MMA output, so it aligns lane-for-lane (both index the
//! accumulator frag's `lane_rc`); the combine `score = c_sq − 2·cross` is then a
//! pair of per-lane f32 elementwise ops. Arch-portable (gfx942 wave64 / gfx1151
//! wave32) via the role-based fragment shortcuts — no hardcoded fragment.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{ConstValue, UOp};
use svod_tensor::Tensor;

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
/// corpus / query / D dims must each be a multiple of it. Also the query width and
/// the top-K slot padding (`K_pad`). The corpus stream tile height is [`TM`].
const BLK: usize = 16;

/// The corpus stream-tile height for Stage 2 ([`build_knn_topk`]) — a multiple of
/// [`BLK`] (16). The corpus is streamed in `M/TM` tiles, so the running top-K runs
/// `M/TM × k` insert passes instead of `M/16 × k`: a taller `TM` trades fewer insert
/// passes for a proportionally heavier per-step reduce. Each `[TM, query]` score
/// sub-tile is `TM/16` stacked 16-row WMMA fragments; `row_arg_reduce` folds across
/// those stacked frags INTERNALLY, returning one logical-`TM`-row argmin per query
/// (the in-primitive cross-frag fold). MUST be a multiple of 16; independent of the
/// query width and `K_pad`
/// (both stay [`BLK`]). `32` is the measured gfx1151 optimum (≈9–22% faster than the
/// `BLK`-tall stream, the win growing with `M`); `64` over-grows the per-step cost
/// and regresses, so the sweet spot is `2·BLK`, not maximal.
const TM: usize = 32;

/// The GPU arch(es) this kernel is built for: gfx942 (CDNA3 MFMA, wave64) and
/// gfx1151 (RDNA3.5 WMMA, wave32). Both resolve the accumulator/operand fragments
/// by role through [`crate::ArchCaps`]; the launcher gates against this list.
/// Validated on gfx942 (CDNA3) and gfx1151 (RDNA3.5).
pub const KNN_SUPPORTED_ARCHS: crate::ArchSet =
    crate::ArchSet::amd(&[svod_dtype::AmdArch::Gfx942, svod_dtype::AmdArch::Gfx1151]);

const POS_INF: f64 = f64::INFINITY;
const NEG_INF: f64 = f64::NEG_INFINITY;

fn fconst(dt: &DType, v: f64) -> Arc<UOp> {
    UOp::const_(dt.clone(), ConstValue::Float(v))
}
fn iconst32(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(v))
}

/// Anchor a register tile's next op on the corpus-loop range carried by `m_blk`:
/// `t.after([m_blk])` when `m_blk` is the rolled loop index (`Idx::Uop`), a no-op for
/// a `Const` tile index (single-tile Stage 1). The rolled-loop re-init footgun
/// ([`crate::loop_scope`]): a constant fill with no loop dependency hoists to
/// `run_count = 1`.
fn reinit_on<'k>(t: RT<'k>, m_blk: &Idx) -> RT<'k> {
    match m_blk {
        Idx::Uop(u) => t.after(u),
        Idx::Const(_) => t,
    }
}

/// The cross-term + `c_sq` combine yielding one `[m_rows, query]` Col f32 score tile
/// `score[m, n] = ‖c[m]‖² − 2·⟨x[n], c[m]⟩` for corpus rows `[m_blk·m_rows, +m_rows)`
/// — the Stage-1 machinery, factored so Stage 2 calls it per [`TM`]-tall corpus tile
/// (and [`build_knn_score`] still uses it for the whole corpus in one tile).
/// `m_rows` is the corpus-tile height (the full `corpus` for Stage 1, [`BLK`] per
/// stream tile for Stage 2). `x_reg_t` is the loop-invariant query operand `[d,
/// query]` (the caller loads it once); `m_blk` is the corpus tile index (its
/// row-block offset, in units of `m_rows`). `masked` gates the GLOBAL→LDS/REG hops
/// against the true corpus extent so a ragged final tile reads `0.0` instead of
/// touching out-of-bounds memory (the caller then masks those rows to `+∞` for the
/// argmin).
#[allow(clippy::too_many_arguments)]
fn score_tile<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    m_rows: usize,
    query: usize,
    d: usize,
    x_reg_t: &RT<'k>,
    c_gl: &GL,
    c_sq_gl: &GL,
    m_blk: &Idx,
    masked: bool,
) -> RT<'k> {
    let bf16 = DType::BFloat16;
    let (row, col) = (TileLayout::Row, TileLayout::Col);

    // GLOBAL(corpus tile m_blk) → LDS (swizzled) → REG, transposed to the `[d, m_rows]`
    // Col operand for the contraction over D (mirrors `fa_qk`'s Kᵀ).
    let c_smem = ker.shared_sw((m_rows, d), bf16.clone(), row);
    let c_reg = ker.operand((m_rows, d), bf16.clone(), row);
    let c_reg_t = ker.operand((d, m_rows), bf16.clone(), col);

    let c_smem = warp.load(c_smem, c_gl.clone(), MoveIdx::block((0, 0, m_blk.clone(), 0), 2));
    let c_reg = warp.load(c_reg, c_smem, MoveIdx::default());
    let c_reg_t = warp.transpose(c_reg_t, &c_reg);

    // score[m, n] = Σ_d c[m,d]·x[n,d] = ⟨c[m], x[n]⟩ (corpus = row, query = col).
    // The MMA accumulator must be RE-ZEROED each corpus iteration: when `m_blk` is the
    // rolled corpus-loop index, anchor the zero-fill on it (`cross.after([loop_range])`,
    // exactly `fa_qk`'s `warp.zero(lp.reinit(att))`) or the constant fill hoists out of
    // the loop (`run_count = 1`) and the MMA accumulates the cross term across ALL tiles.
    // A `Const` `m_blk` (the single-tile Stage-1 path) adds no dependency.
    let cross = warp.zero(reinit_on(ker.acc((m_rows, query), col), m_blk));
    let cross = warp.mma_atb(cross, &c_reg_t, x_reg_t);

    // Load c_sq_rep[m_blk] into the SAME accumulator fragment + layout as the cross
    // MMA output, so the two align lane-for-lane (both index the accumulator frag's
    // `lane_rc`; orientation-robust per the reductions/masked tests).
    let cs_mi = MoveIdx::block((0, 0, m_blk.clone(), 0), 2);
    let cs_mi = if masked { cs_mi.masked() } else { cs_mi };
    let c_sq = warp.load(ker.acc((m_rows, query), col), c_sq_gl.clone(), cs_mi);

    // score = c_sq − 2·cross, all f32.
    let cross = warp.mul_scalar(cross, -2.0);
    warp.add(cross, &c_sq)
}

/// Build the x²-free KNN score-tile kernel into the bound ABI.
///
/// ABI (outputs then inputs, fixed by [`Kernel::bind_abi`]):
/// - `score` (`[1, 1, corpus, query]`, f32) — the output `‖c[m]‖² − 2·⟨x[n],c[m]⟩`.
/// - `x` (`[1, 1, query, d]`, bf16) — the query rows.
/// - `c` (`[1, 1, corpus, d]`, bf16) — the corpus rows.
/// - `c_sq_rep` (`[1, 1, corpus, query]`, f32) — `‖c[m]‖²` precomputed outside the
///   kernel and replicated along the query axis (each `(m, n)` holds `c_sq[m]`).
///
/// Single-warp; `corpus`, `query`, `d` must each be a multiple of [`BLK`] (16).
///
/// # Panics
/// Panics unless `corpus`, `query`, and `d` are each a multiple of 16.
pub fn build_knn_score(ker: &Kernel, corpus: usize, query: usize, d: usize) {
    Kernel::assert_divisible(corpus, BLK, "KNN corpus");
    Kernel::assert_divisible(query, BLK, "KNN query");
    Kernel::assert_divisible(d, BLK, "KNN D");

    let bf16 = DType::BFloat16;
    let f32 = DType::Float32;
    let warp = ker.warp();

    // ABI: output (score, f32) then inputs (x, c — bf16; c_sq_rep — f32).
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[1, 1, corpus, query], f32.clone())],
        &[
            GlSpec::new(&[1, 1, query, d], bf16.clone()),
            GlSpec::new(&[1, 1, corpus, d], bf16.clone()),
            GlSpec::new(&[1, 1, corpus, query], f32.clone()),
        ],
    );
    let (score_gl, x_gl, c_gl, c_sq_gl): (GL, GL, GL, GL) =
        (outs[0].clone(), ins[0].clone(), ins[1].clone(), ins[2].clone());

    // Query tile loaded once and transposed to its `[d, query]` Col fragment.
    let x_reg_t = load_query_t(ker, &warp, query, d, &x_gl, &Idx::Const(0));

    // The whole corpus in one `(corpus, query)` tile (the Stage-1 single-store shape).
    let score = score_tile(ker, &warp, corpus, query, d, &x_reg_t, &c_gl, &c_sq_gl, &Idx::Const(0), false);
    let _ = warp.store(score_gl, score, MoveIdx::block((0, 0, 0, 0), 2));
}

/// Load the query tile `[query, d]` and transpose it to its `[d, query]` Col
/// operand fragment for the cross contraction over D — loop-invariant, so both
/// builders load it once. `q_blk` is the query-block index this workgroup owns (in
/// query-tile-height units): the GLOBAL→LDS load offsets axis 2 by it, so the grid's
/// `block_idx[0]` selects each 16-query block of a wide `[Npad, d]` query input
/// (block-unit offset ⇒ the element offset is `q_blk·query`, exactly `q_blk·16`).
/// `Idx::Const(0)` is the single-block (Stage-1 / `N ≤ 16`) path.
fn load_query_t<'k>(ker: &'k Kernel, warp: &Group<'k>, query: usize, d: usize, x_gl: &GL, q_blk: &Idx) -> RT<'k> {
    let bf16 = DType::BFloat16;
    let (row, col) = (TileLayout::Row, TileLayout::Col);
    let x_smem = ker.shared_sw((query, d), bf16.clone(), row);
    let x_reg = ker.operand((query, d), bf16.clone(), row);
    let x_reg_t = ker.operand((d, query), bf16.clone(), col);
    let x_smem = warp.load(x_smem, x_gl.clone(), MoveIdx::block((0, 0, q_blk.clone(), 0), 2));
    let x_reg = warp.load(x_reg, x_smem, MoveIdx::default());
    warp.transpose(x_reg_t, &x_reg)
}

/// Per-query running top-K state: two Col-layout `[K_pad=BLK, query]` register
/// tiles — `val` (f32, K-slot = row, seeded `+∞`) and `idx` (Int32, seeded `−1`).
/// `row_arg_reduce` folds the K-slot (row) axis on both archs, so a `Max` reduce
/// yields the per-query running-worst slot to evict.
struct TopK<'k> {
    val: RT<'k>,
    idx: RT<'k>,
}

/// Build the x²-free KNN running-top-K kernel into the bound ABI.
///
/// ABI (outputs then inputs):
/// - `idx` (`[1, 1, query, k]`, Int32) — the **unsorted** K nearest corpus indices
///   per query (final K-ordering offloaded to the generic graph in Stage 3).
/// - `val` (`[1, 1, query, k]`, f32) — their x²-free scores.
/// - `x` (`[1, 1, query, d]`, bf16) — the query rows.
/// - `c` (`[1, 1, corpus, d]`, bf16) — the corpus rows.
/// - `c_sq_rep` (`[1, 1, corpus, query]`, f32) — `‖c[m]‖²` replicated along query.
///
/// Single-warp, correctness-first, arch-portable via role fragments. The corpus is
/// streamed in [`TM`]-tall tiles through a [`crate::loop_scope::Loop`]; per tile
/// the running top-K is updated by up to `k` argmin-insert steps. Built **rolled**
/// (`arg_reduce` panics under unroll).
///
/// **Query-block grid tiling:** each workgroup processes ONE `query`(= [`BLK`]) block,
/// selected by `block_idx[0]` — the grid is `[ceil(Npad/16), 1, 1]`, so it covers a
/// wide `[Npad, *]` query input. The block index offsets ONLY the query (x) load and
/// the output store (query-independent steps — the corpus stream, score, `c_sq` load,
/// argmin-insert — are block-relative); a `[1,1,1]` grid ⇒ block 0 ⇒ the single-block
/// path. The `[1,1,query,*]` x/output globals address the wider real buffers because
/// the offset rides the (identical) row stride, not the declared extent.
///
/// # Panics
/// Panics unless `query`/`d` are multiples of [`BLK`], `corpus > 0`, `1 ≤ k ≤ BLK`,
/// and `query ≤ BLK` (the v1 single-query-fragment constraint: a wider query would
/// fold distinct queries together in the per-query `row_arg_reduce`).
pub fn build_knn_topk(ker: &Kernel, corpus: usize, query: usize, d: usize, k: usize) {
    Kernel::assert_divisible(query, BLK, "KNN topk query");
    Kernel::assert_divisible(d, BLK, "KNN topk D");
    Kernel::assert_divisible(TM, BLK, "KNN topk TM");
    assert!(corpus > 0, "KNN topk corpus must be > 0");
    assert!((1..=BLK).contains(&k), "KNN topk k must be in 1..=16");
    assert!(query <= BLK, "KNN topk query must be <= 16 (single query fragment) for v1");

    let bf16 = DType::BFloat16;
    let f32 = DType::Float32;
    let i32 = DType::Int32;
    let col = TileLayout::Col;
    let warp = ker.warp();
    let acc_frag = ker.frag(FragRole::Accumulator);

    // ABI: outputs (idx i32, val f32) then inputs (x, c — bf16; c_sq_rep — f32).
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[1, 1, query, k], i32.clone()), GlSpec::new(&[1, 1, query, k], f32.clone())],
        &[
            GlSpec::new(&[1, 1, query, d], bf16.clone()),
            GlSpec::new(&[1, 1, corpus, d], bf16.clone()),
            GlSpec::new(&[1, 1, corpus, query], f32.clone()),
        ],
    );
    let (idx_gl, val_gl, x_gl, c_gl, c_sq_gl): (GL, GL, GL, GL, GL) =
        (outs[0].clone(), outs[1].clone(), ins[0].clone(), ins[1].clone(), ins[2].clone());

    // Query-block grid tiling: each workgroup owns one 16-query block, selected by
    // `block_idx[0]`. The grid is `[ceil(Npad/16), 1, 1]`, so the offset rides ONLY the
    // query (x) load and the output store — the corpus stream, score, c_sq load (all
    // query-independent), and the per-query argmin-insert are block-relative and
    // unchanged. A `[1,1,1]` grid ⇒ block 0 ⇒ the single-query-block path.
    let q_blk = Idx::Uop(ker.block_idx[0].clone());
    let x_reg_t = load_query_t(ker, &warp, query, d, &x_gl, &q_blk);

    // Running top-K state (Col `[K_pad=BLK, query]`). The fragment is 16-wide but only
    // the first `k` K-slots are live; seed slots `[0, k)` to `+∞` (empty, fillable) and
    // the padding `[k, 16)` to `−∞` so the `row_arg_reduce(Max)` worst-slot search NEVER
    // evicts into a padding slot (a `−∞` always loses the Max to a real `+∞`/finite
    // slot). Without this the worst is forever the `+∞` of an unused slot and inserts
    // leak past the stored first-`k` columns. `idx` seeds to `−1` everywhere.
    let val0 = seed_topk_val(ker, &warp, query, k);
    let idx0 = warp.map(ker.rt((BLK, query), i32.clone(), col, acc_frag), |_, _| iconst32(-1));
    let topk = TopK { val: val0, idx: idx0 };

    // Stream the corpus in TM-tall tiles via the FA running-state Loop carry.
    let tiles = corpus.div_ceil(TM);
    let masked = !corpus.is_multiple_of(TM);
    let lp = ker.loop_static(tiles as i64);
    let m_tile = lp.index().clone();
    let topk = TopK { val: lp.reinit(topk.val), idx: lp.reinit(topk.idx) };

    let topk = topk_insert(ker, &warp, corpus, query, d, k, &x_reg_t, &c_gl, &c_sq_gl, &m_tile, masked, topk);

    // Close the loop once: `topk_insert` chained the idx-evict store (the last
    // terminal) to depend on the val-evict store and the score updates, so the single
    // loop-closing END scopes the whole insert body (the matmul multi-accumulator
    // idiom). Both carried tiles then read their post-loop value via `.after([end])`.
    let ended = lp.close();
    let idx_after = topk.idx.after(&ended);
    let val_after = topk.val.after(&ended);

    store_topk(ker, &warp, query, k, &idx_gl, &val_gl, &idx_after, &val_after, &q_blk);
}

/// One corpus tile's argmin-insert: compute the score sub-tile, mask ragged rows
/// to `+∞`, then run up to `k` steps of (find the per-query tile-min over corpus,
/// compare to the running worst slot, conditionally evict, remove the consumed
/// element). Returns the updated running top-K.
#[allow(clippy::too_many_arguments)]
fn topk_insert<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    corpus: usize,
    query: usize,
    d: usize,
    k: usize,
    x_reg_t: &RT<'k>,
    c_gl: &GL,
    c_sq_gl: &GL,
    m_tile: &Arc<UOp>,
    masked: bool,
    mut topk: TopK<'k>,
) -> TopK<'k> {
    // score[m, n] for this corpus tile — `TM`-tall (`TM/16` stacked 16-row frags),
    // corpus = row, query = col.
    let mut score = score_tile(ker, warp, TM, query, d, x_reg_t, c_gl, c_sq_gl, &Idx::Uop(m_tile.clone()), masked);
    if masked {
        score = mask_ragged_rows(warp, score, m_tile, corpus);
    }

    // Each step's stores are CHAINED so a single loop-closing END (in the caller)
    // scopes the whole insert body inside the rolled corpus loop (the matmul
    // multi-accumulator idiom): idx-evict ← val-evict, score-remove ← idx-evict, and
    // the next step's reduces read the chained `score`/`topk.val`. The idx-evict of
    // the LAST step is the loop's terminal store, so its `remove_used` (dead — no next
    // argmin) is skipped, leaving idx-evict last on the store stack for `lp.close()`.
    for step in 0..k {
        // a. per-query tile-min over the TM corpus rows. `row_arg_reduce` folds the
        //    `[TM, query]` Col score's `TM/16` stacked height-frags INTERNALLY (the
        //    in-primitive cross-frag fold), returning one `(row_min, row_arg)` per
        //    query directly — `row_arg` is the in-tile corpus row `0..TM` (global
        //    over the stacked frags). `global_m = m_tile·TM + row_arg`.
        let (row_min, row_arg) =
            warp.row_arg_reduce(seed_val(ker, warp, BLK, POS_INF), seed_idx(ker, warp, BLK), &score, ArgDir::Min);
        // `global_m = m_tile·TM + row_arg` in a FRESH RV — `warp.map` rewrites its
        // tile in place, so mapping `row_arg` directly would clobber the in-tile index
        // that `remove_used` still needs to mask the consumed element (the `k > 1`
        // per-step argmin-skip). Read `row_arg` (anchored) into the new buffer instead.
        let mbase = m_tile.mul(&cidx(TM as i64)).cast(DType::Int32);
        let (ra_buf, ra_shape) = (warp.anchor(row_arg.uop()), row_arg.shape().to_vec());
        let global_m = warp.map(seed_idx(ker, warp, BLK), move |_, idx| {
            load_at(&ra_buf, &ra_shape, idx).try_add(&mbase).expect("global_m")
        });

        // b. per-query running-worst value + its K-slot index. The running top-K is
        //    `[K_pad=BLK, query]` (one frag), so this reduce needs no fold.
        let (worst, evict) =
            warp.row_arg_reduce(seed_val(ker, warp, BLK, NEG_INF), seed_idx(ker, warp, BLK), &topk.val, ArgDir::Max);

        // d. Evict (conditional rewrite by K-slot): write row_min/global_m into the
        //    `evict[query]` K-slot where `row_min[query] < worst[query]` (do_insert).
        //    Chain idx-evict after val-evict so both carried writes share one END.
        topk.val = evict_slot(warp, topk.val, &evict, &row_min, &worst, &row_min);
        topk.idx = evict_slot(warp, topk.idx.after(&topk.val), &evict, &row_min, &worst, &global_m);

        // e. Remove the consumed corpus element so the next step's argmin skips it
        //    (chained after idx-evict). Skipped on the last step (no next argmin), so
        //    idx-evict stays the loop's terminal store.
        if step + 1 < k {
            score = remove_used(warp, score.after(&topk.idx), &row_arg, &row_min, &worst);
        }
    }
    topk
}

/// A length-`length` f32 RV seeded to `init` (the `row_arg_reduce` value
/// accumulator; it actually overwrites the seed with `dir.init()`, but a same-dtype
/// seed keeps the alloc explicit). `length` is always `BLK` (one output slot per
/// lane): `row_arg_reduce` collapses the whole reduced axis — including the score
/// tile's `TM/16` stacked frags, folded internally — into that single slot.
fn seed_val<'k>(ker: &'k Kernel, warp: &Group<'k>, length: usize, init: f64) -> RV<'k> {
    let frag = ker.frag(FragRole::Accumulator);
    warp.clear_rv(ker.rv(length, DType::Float32, VecLayout::Ortho, frag), init)
}
/// A length-`length` Int32 index RV seeded to `−1` (the `row_arg_reduce` index acc).
fn seed_idx<'k>(ker: &'k Kernel, warp: &Group<'k>, length: usize) -> RV<'k> {
    let frag = ker.frag(FragRole::Accumulator);
    warp.clear_rv(ker.rv(length, DType::Int32, VecLayout::Ortho, frag), -1.0)
}

/// Seed the running-top-K value tile (Col `[K_pad=BLK, query]`): K-slots `[0, k)`
/// to `+∞` (empty, fillable), the padding slots `[k, 16)` to `−∞` so the worst-slot
/// `row_arg_reduce(Max)` never evicts into a padding slot. The per-element K-slot
/// is its global row position, computed arch-correctly via [`Group::map_position`].
/// Both branches are constant (`x` is only used for its dtype), so this uses the
/// seed form directly instead of [`Group::mask_where`].
fn seed_topk_val<'k>(ker: &'k Kernel, warp: &Group<'k>, query: usize, k: usize) -> RT<'k> {
    let tile = ker.acc((BLK, query), TileLayout::Col);
    let k_c = cidx(k as i64);
    warp.map_position(tile, Idx::Const(0), Idx::Const(0), move |x, _idx, k_pos, _q_pos| {
        UOp::try_where(k_pos.lt(&k_c), fconst(&x.dtype(), POS_INF), fconst(&x.dtype(), NEG_INF))
            .expect("topk val seed where")
    })
}

/// Read a per-query RV scalar inside a `map` over a Col `[*, query]` tile: the
/// query is the tile's column / width axis, so it selects RV slot `idx[1]` exactly
/// as [`crate::math`]'s `combine_rv` does for a Col tile. The RV buffer is anchored
/// so a constant-address read of a carried (loop) RV is not hoisted out of the
/// corpus loop. Returns `(anchored_buf, shape)` to capture into the `map` closure.
fn rv_query_src<'k>(warp: &Group<'k>, rv: &RV<'k>) -> (Arc<UOp>, Vec<usize>) {
    (warp.anchor(rv.uop()), rv.shape().to_vec())
}

/// Mask ragged corpus rows (`global_m ≥ corpus`) of a Col `[TM, query]` score
/// tile to `+∞` via [`Group::mask_where`], so the per-query argmin never selects
/// the padding the masked score load zeroed. The per-element corpus row
/// (`global_m = m_tile·TM + idx[0]·16 + m_if`) is computed arch-correctly inside
/// `mask_where` — the `m_tile` block offset threads the stream stride.
fn mask_ragged_rows<'k>(warp: &Group<'k>, score: RT<'k>, m_tile: &Arc<UOp>, corpus: usize) -> RT<'k> {
    let bound = cidx(corpus as i64);
    warp.mask_where(score, Idx::Uop(m_tile.clone()), Idx::Const(0), POS_INF, move |global_m, _| global_m.ge(&bound))
}

/// Evict step (used for BOTH the f32 value tile and its Int32 index partner): in
/// the Col `[BLK, query]` tile, overwrite the element at the per-query worst K-slot
/// `evict[query]` with `repl[query]` when `do_insert = row_min[query] <
/// worst[query]`, leaving every other slot. Selecting both outputs (value tile and
/// index tile) by the SAME predicate keeps the kept value paired with its index.
/// The K-slot of an element is its global row position (computed via
/// [`Group::map_position`]); the per-query RVs are read by the col-tile index
/// (`idx[1]`), the multi-RV generalization of `combine_rv`.
#[allow(clippy::too_many_arguments)]
fn evict_slot<'k>(
    warp: &Group<'k>,
    tile: RT<'k>,
    evict: &RV<'k>,
    row_min: &RV<'k>,
    worst: &RV<'k>,
    repl: &RV<'k>,
) -> RT<'k> {
    let (e_buf, e_shape) = rv_query_src(warp, evict);
    let (rmin_buf, rmin_shape) = rv_query_src(warp, row_min);
    let (worst_buf, worst_shape) = rv_query_src(warp, worst);
    let (repl_buf, repl_shape) = rv_query_src(warp, repl);
    warp.map_position(tile, Idx::Const(0), Idx::Const(0), move |x, idx, k_pos, _col| {
        let k_pos = k_pos.cast(DType::Int32);
        let q = idx[1].clone();
        let e = load_at(&e_buf, &e_shape, &[q.clone(), Idx::Const(0)]);
        let rmin = load_at(&rmin_buf, &rmin_shape, &[q.clone(), Idx::Const(0)]);
        let wst = load_at(&worst_buf, &worst_shape, &[q.clone(), Idx::Const(0)]);
        let mut rpl = load_at(&repl_buf, &repl_shape, &[q, Idx::Const(0)]);
        if rpl.dtype() != x.dtype() {
            rpl = rpl.cast(x.dtype());
        }
        let do_insert = rmin.lt(&wst);
        let hit = k_pos.eq(&e).and_(&do_insert);
        UOp::try_where(hit, rpl, x.clone()).expect("evict where")
    })
}

/// Remove the consumed corpus element from a Col `[TM, query]` score tile: set
/// `score[m == row_arg[query], query] = +∞` where `row_min[query] < worst[query]`
/// (i.e. the element actually inserted this step), so the next step's argmin skips
/// it. `row_arg` is the in-tile corpus row (0..TM) `row_arg_reduce` folded across
/// the stacked frags, compared against the element's global row position (computed
/// via [`Group::map_position`]). Exactly one tile position matches (the frag
/// decomposition `0..TM ↔ (idx[0], in-frag row)` is a bijection), so no
/// double-removal.
fn remove_used<'k>(warp: &Group<'k>, score: RT<'k>, row_arg: &RV<'k>, row_min: &RV<'k>, worst: &RV<'k>) -> RT<'k> {
    let (ra_buf, ra_shape) = rv_query_src(warp, row_arg);
    let (rmin_buf, rmin_shape) = rv_query_src(warp, row_min);
    let (worst_buf, worst_shape) = rv_query_src(warp, worst);
    warp.map_position(score, Idx::Const(0), Idx::Const(0), move |x, idx, m_local, _col| {
        let m_local = m_local.cast(DType::Int32);
        let q = idx[1].clone();
        let ra = load_at(&ra_buf, &ra_shape, &[q.clone(), Idx::Const(0)]);
        let rmin = load_at(&rmin_buf, &rmin_shape, &[q.clone(), Idx::Const(0)]);
        let wst = load_at(&worst_buf, &worst_shape, &[q, Idx::Const(0)]);
        let do_insert = rmin.lt(&wst);
        let hit = m_local.eq(&ra).and_(&do_insert);
        UOp::try_where(hit, fconst(&x.dtype(), POS_INF), x.clone()).expect("remove-used where")
    })
}

/// Store the unsorted running top-K to the `[1, 1, query, k]` outputs. The running
/// tiles are Col `[K_slot=BLK, query]`; the output wants `[query, K_slot]`, the
/// transpose. So each tile is `transpose`d into a Row `[query, BLK]`
/// AccumulatorT-fragment tile (the FA output-store relayout) and stored with the
/// boundary mask, which drops the columns `≥ k` (the K-slots past the requested
/// `k`). `k == BLK` needs no mask; a partial `k` gates the trailing columns.
#[allow(clippy::too_many_arguments)]
fn store_topk<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    query: usize,
    k: usize,
    idx_gl: &GL,
    val_gl: &GL,
    idx_after: &RT<'k>,
    val_after: &RT<'k>,
    q_blk: &Idx,
) {
    let row = TileLayout::Row;
    let i32 = DType::Int32;
    let acc_t = ker.frag(FragRole::AccumulatorT);
    // Offset the output store's query-row block by `q_blk` (this workgroup's query
    // block, in query-tile-height units), the store-side mirror of `load_query_t`'s
    // load offset, so each workgroup writes its own 16-query rows of the `[Npad, k]`
    // output. `Idx::Const(0)` ⇒ the single-block (`N ≤ 16`) path.
    let mi = if k.is_multiple_of(BLK) {
        MoveIdx::block((0, 0, q_blk.clone(), 0), 2)
    } else {
        MoveIdx::block((0, 0, q_blk.clone(), 0), 2).masked()
    };

    // Transpose Col [BLK, query] → Row [query, BLK] (AccumulatorT), then masked-store
    // its first `k` columns to the [query, k] output. Both transposes (which push
    // intermediate REG stores) come BEFORE both global stores, so the kernel's final
    // two terminal stores — popped by `finish(2)` as the SINK sources — are exactly
    // the two output writes.
    let val_t = warp.transpose(ker.acc_t((query, BLK), row), val_after);
    let idx_t = warp.transpose(ker.rt((query, BLK), i32.clone(), row, acc_t), idx_after);
    let _ = warp.store(val_gl.clone(), val_t, mi.clone());
    let _ = warp.store(idx_gl.clone(), idx_t, mi);
}

/// Round `x` up to the next multiple of [`BLK`] (16) — the WMMA tile edge the
/// kernel's `D`/query block geometry requires.
fn pad16(x: usize) -> usize {
    x.div_ceil(BLK) * BLK
}

// =============================================================================
// Stage 3 — the public lazy-Tensor KNN entry point + the generic-graph tail.
// =============================================================================

/// **Graph-native** fused brute-force K-nearest-neighbors — the matmul/FA peer for
/// KNN, returning lazy output [`Tensor`]s (the tile kernel is a `custom_kernel` /
/// `Op::Call` node, the K-ordering + exact distances are ordinary generic-graph ops).
///
/// For `N` query rows `x` (`[N, D]`) and `M` corpus rows `c` (`[M, D]`, **any float
/// dtype**) it returns `Some((dists, idxs))`:
/// - `idxs` (`[N, k]`, i32) — the `k` nearest corpus rows per query, **sorted
///   ascending by distance** (ties → smaller corpus index, matching a brute-force
///   reference / [`Tensor::topk`]).
/// - `dists` (`[N, k]`, f32) — their **true** squared-L2 distances
///   `‖x[n] − c[idxs[n,j]]‖²`, recomputed exactly in f32 (the kernel's x²-free score
///   only orders the corpus; the self-term `‖x‖²` is re-added here).
///
/// The kernel streams the corpus and keeps the running top-K from the x²-free score
/// `‖c[m]‖² − 2·⟨x[n],c[m]⟩` ([`build_knn_topk`]); this entry owns the host-side prep
/// (cast → bf16, zero-pad `D`/`N` to the WMMA edge, the f32 `‖c‖²`) and the
/// generic-graph tail (sort the K, gather the sorted corpus rows, exact f32
/// distances). The corpus `M` is NOT padded — the kernel ragged-masks its final tile.
///
/// Like [`crate::matmul`] / [`crate::flash_attention_with`], the outcome is three-way
/// (via [`crate::launch_custom`]):
/// - `Ok(Some((dists, idxs)))` — ran (lazy nodes; `prepare()` to realize).
/// - `Ok(None)` — the device isn't a supported arch ([`KNN_SUPPORTED_ARCHS`] —
///   gfx942 / gfx1151 with the AMD toolchain). The caller substitutes its own KNN.
/// - `Err` — a malformed request on a supported device: `x`/`c` not statically-shaped
///   rank-2 tensors, mismatched `D`, `k > M`, or `k` outside the kernel's `1..=16`.
///   These are caller bugs (a genuine kernel build/dispatch failure also returns `Err`).
///
/// ```no_run
/// use svod_tensor::Tensor;
/// let x = Tensor::randn(&[40, 20]).unwrap(); // 40 queries, dim 20
/// let c = Tensor::randn(&[100, 20]).unwrap(); // 100 corpus rows
/// if let Some((mut dists, mut idxs)) = svod_tk::knn(&x, &c, 5).unwrap() {
///     dists.prepare().unwrap(); // [40, 5] f32 squared-L2 to the 5 nearest
///     idxs.prepare().unwrap();  // [40, 5] i32 corpus indices (ascending by distance)
/// }
/// ```
pub fn knn(x: &Tensor, c: &Tensor, k: usize) -> crate::LaunchResult<Option<(Tensor, Tensor)>> {
    use snafu::{ResultExt, ensure};

    let xd = crate::launch::concrete_dims(x, "knn", "x", 2)?;
    let cd = crate::launch::concrete_dims(c, "knn", "c", 2)?;
    let (n, dx) = (xd[0], xd[1]);
    let (m, dc) = (cd[0], cd[1]);

    // Structural validity (`Err`) — checked BEFORE arch resolution, like `concrete_dims`:
    // D mismatch and the k bounds are FIXED request properties, so a violation is a
    // caller bug regardless of the device (never silently `None`).
    ensure!(dx == dc, crate::launch::OperandDimMismatchSnafu { kernel: "knn", dim: "D", a: dx, b: dc });
    ensure!(
        (1..=BLK).contains(&k),
        crate::launch::DimMultipleSnafu { kernel: "knn", dim: "k (must be 1..=16)", value: k, multiple: 1usize }
    );
    ensure!(
        k <= m,
        crate::launch::DimMultipleSnafu { kernel: "knn", dim: "k (must be <= corpus M)", value: k, multiple: m }
    );

    // The three-way policy (cf. `launch_custom`), inlined because the build yields a
    // tuple (`launch_custom` is single-Tensor): `None` for the wrong arch/toolchain
    // (caller's fallback), `Err` for a malformed request (handled above), `Some` when run.
    let Some(arch) = crate::target::resolve_supported_arch(&x.device(), KNN_SUPPORTED_ARCHS).ok() else {
        return Ok(None);
    };

    let caps = crate::ArchCaps::for_arch(arch);
    let (f32, bf16) = (DType::Float32, DType::BFloat16);
    let d_pad = pad16(dx);
    let n_pad = pad16(n);

    // f32 copies for the exact-distance tail (corpus stays unpadded — the tail gathers
    // TRUE D-rows; the query keeps its N rows).
    let x_f32 = x.cast(f32.clone()).context(crate::launch::OperandSnafu)?;
    let c_f32 = c.cast(f32.clone()).context(crate::launch::OperandSnafu)?;

    // Kernel bf16 operands, zero-padded to the WMMA edge. Zeros contribute 0 to ⟨x,c⟩
    // and to ‖c‖², so the score is unchanged; padded query rows produce junk top-Ks the
    // tail slices off. `try_pad` pads with zeros.
    let x_bf = pad_operand(&x.cast(bf16.clone()).context(crate::launch::OperandSnafu)?, n, dx, n_pad, d_pad)?;
    let c_bf = pad_operand(&c.cast(bf16.clone()).context(crate::launch::OperandSnafu)?, m, dc, m, d_pad)?;

    // c_sq[m] = Σ_d c[m,d]² in f32 (query-independent), replicated to the kernel's
    // [1,1,M,BLK] (one query-block width — every query block reads the same slice).
    let c_sq_rep = c_sq_replicated(&c_f32, m)?;

    let idx_t = Tensor::empty(&[1, 1, n_pad, k], DType::Int32);
    let val_t = Tensor::empty(&[1, 1, n_pad, k], f32.clone());
    let grid = [(n_pad / BLK) as i64, 1, 1];
    let block = caps.wave_size as i64;

    // The kernel processes ONE 16-query block per workgroup (`query = BLK`);
    // `block_idx[0]` selects the block, so its declared `[1,1,BLK,*]` x/output globals
    // address the wider real `[1,1,Npad,*]` buffers (identical row stride).
    let outs = crate::graph_launch_multi(
        "knn_topk",
        grid,
        block,
        vec![idx_t, val_t],
        &[&x_bf, &c_bf, &c_sq_rep],
        caps,
        move |ker| {
            build_knn_topk(ker, m, BLK, d_pad, k);
            ker.finish(2)
        },
    )?;
    let (idx_raw, val_raw) = (outs[0].clone(), outs[1].clone());

    knn_tail(&idx_raw, &val_raw, &x_f32, &c_f32, n, dx, k).map(Some)
}

/// Zero-pad a `[rows, d]` tensor's last (`D`) axis to `d_pad` and its leading (row)
/// axis to `rows_pad`, then add the kernel's `[1, 1, …]` leading singleton axes —
/// the bf16 kernel operand layout. Zeros are the additive identity in ⟨x,c⟩/‖c‖², so
/// the padding leaves the score unchanged; padded query rows are sliced off in the tail.
fn pad_operand(t: &Tensor, rows: usize, d: usize, rows_pad: usize, d_pad: usize) -> crate::LaunchResult<Tensor> {
    use snafu::ResultExt;
    let padded = t
        .try_pad(&[(0, (rows_pad - rows) as isize), (0, (d_pad - d) as isize)])
        .context(crate::launch::OperandSnafu)?;
    padded.try_reshape([1isize, 1, rows_pad as isize, d_pad as isize]).context(crate::launch::OperandSnafu)
}

/// `c_sq[m] = Σ_d c_f32[m,d]²` in f32, replicated to the kernel's `[1, 1, M, BLK]`
/// `c_sq_rep` operand (each `(m, n)` reads `c_sq[m]`). `c_sq` is query-independent, so
/// one query-block width (`BLK = 16`) suffices regardless of `N` — every query block
/// reads the same slice.
fn c_sq_replicated(c_f32: &Tensor, m: usize) -> crate::LaunchResult<Tensor> {
    use snafu::ResultExt;
    let c_sq = c_f32
        .try_mul(c_f32)
        .context(crate::launch::OperandSnafu)?
        .sum_with()
        .axes(1isize)
        .keepdim(true)
        .call()
        .context(crate::launch::OperandSnafu)?; // [M, 1]
    c_sq.try_reshape([1isize, 1, m as isize, 1])
        .context(crate::launch::OperandSnafu)?
        .try_expand([1isize, 1, m as isize, BLK as isize])
        .context(crate::launch::OperandSnafu)
}

/// The generic-graph tail over the kernel's UNSORTED top-K (`idx_raw`/`val_raw`,
/// `[1,1,Npad,k]`): slice off the padded query rows, sort the `k` per query ascending
/// by the x²-free score (its order equals the true-distance order — `‖x‖²` is constant
/// per query), gather the sorted corpus rows, and recompute the EXACT f32 squared-L2.
/// Returns `(dists [N,k] f32, idx_sorted [N,k] i32)`.
fn knn_tail(
    idx_raw: &Tensor,
    val_raw: &Tensor,
    x_f32: &Tensor,
    c_f32: &Tensor,
    n: usize,
    d: usize,
    k: usize,
) -> crate::LaunchResult<(Tensor, Tensor)> {
    use snafu::ResultExt;
    // Each tensor-op `?` boxes into the launch `Error` (`OperandSnafu`) inline — the
    // launch enum keeps its sources boxed (`clippy::result_large_err`), so the tail
    // never surfaces the large `svod_tensor` Result.
    let op = crate::launch::OperandSnafu;

    // 1. [1,1,Npad,k] → [Npad,k] → [N,k] (drop the padded-query rows).
    let idx = idx_raw
        .try_reshape([-1, k as isize])
        .context(op)?
        .try_shrink([(0, n as isize), (0, k as isize)])
        .context(op)?;
    let val = val_raw
        .try_reshape([-1, k as isize])
        .context(op)?
        .try_shrink([(0, n as isize), (0, k as isize)])
        .context(op)?;

    // 2. Sort the k per query ascending by the x²-free score; reorder the indices to
    //    match. The score order == the true-distance order (‖x‖² is a per-query const).
    let (_val_sorted, perm) = val.sort(1, false).context(op)?;
    let idx_sorted = idx.gather(1, &perm).context(op)?;

    // 3. Exact f32 distances for the sorted indices: gather the TRUE (unpadded) corpus
    //    rows c_f32[idx_sorted] → [N, k, D], then ‖x[n] − c_gathered‖² over D.
    let idx_flat = idx_sorted.try_reshape([(n * k) as isize]).context(op)?;
    let c_gathered =
        c_f32.index_select(0, &idx_flat).context(op)?.try_reshape([n as isize, k as isize, d as isize]).context(op)?;
    let diff = x_f32.try_reshape([n as isize, 1, d as isize]).context(op)?.try_sub(&c_gathered).context(op)?;
    let dists = diff.try_mul(&diff).context(op)?.sum_with().axes(2isize).dtype(DType::Float32).call().context(op)?;

    Ok((dists, idx_sorted))
}
