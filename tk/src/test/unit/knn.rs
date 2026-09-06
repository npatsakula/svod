//! Tests for the x²-free KNN score-tile kernel ([`crate::kernels::knn`]): a
//! GPU-free graph-shape check (both archs) of the cross WMMA + `c_sq` load + f32
//! combine, and a hardware-gated end-to-end score check vs a generic-Tensor
//! reference (arch-portable: gfx942 wave64 / gfx1151 wave32).

use std::sync::Arc;

use svod_dtype::{AmdArch, DType, DeviceSpec};
use svod_ir::{Op, UOp};

use crate::kernels::knn::{KNN_SUPPORTED_ARCHS, build_knn_score, build_knn_topk};
use crate::{ArchCaps, Kernel};
use svod_ir::ops;

/// Placeholder buffers for a GPU-free build: `score` (f32), `x`/`c` (bf16),
/// `c_sq_rep` (f32), in ABI order.
fn knn_bufs(corpus: usize, query: usize, d: usize) -> Vec<Arc<UOp>> {
    vec![
        UOp::new_buffer(DeviceSpec::Cpu, corpus * query, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, query * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, corpus * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, corpus * query, DType::Float32),
    ]
}

/// Build the KNN score SINK for `(corpus, query, d)` on `caps` (GPU-free).
fn knn_sink(corpus: usize, query: usize, d: usize, caps: ArchCaps) -> Arc<UOp> {
    let ker = Kernel::new("knn_score", [1, 1, 1], caps.wave_size as i64, knn_bufs(corpus, query, d), caps);
    build_knn_score(&ker, corpus, query, d);
    ker.finish(1)
}

/// The score kernel's graph carries: a cross-term `WMMA`, a `Load` of the f32
/// `c_sq` input (the 4th / last `Param`) into the accumulator fragment, and the
/// f32 combine (the `−2·cross` `Mul` by a const + the `+ c_sq` `Add`). No stray
/// LDS / barrier surprises beyond the two operand strips the cross MMA needs.
/// Holds on both wave64 (gfx942) and wave32 (gfx1151).
#[test]
fn test_knn_score_graph_shape() {
    for caps in [ArchCaps::GFX942, ArchCaps::for_amd(AmdArch::Gfx1151)] {
        let arch = caps.arch;
        let topo = knn_sink(32, 32, 32, caps).toposort();

        // The cross term emits a WMMA (looped: one symbolic node per K-iteration).
        assert!(topo.iter().any(|u| matches!(u.op(), Op::Wmma(..))), "{arch:?}: cross term emits a WMMA");

        // The c_sq input is the last (4th) Param; its load reads PARAM slot 3.
        let loads_param3 = topo.iter().any(|u| {
            let Op::Load(..) = u.op() else { return false };
            u.toposort().iter().any(|s| matches!(s.op(), Op::Param(ops::Param { arg, .. }) if arg.slot == 3))
        });
        assert!(loads_param3, "{arch:?}: a Load reads the c_sq input (Param slot 3)");

        // The f32 combine: a `*(-2)` multiply against a const and an add. The score
        // path is entirely f32 (no cast back to a narrow dtype before the store).
        let has_neg2 = topo.iter().any(|u| {
            let Op::Binary(svod_ir::BinaryOp::Mul, _, rhs) = u.op() else { return false };
            matches!(rhs.op(), Op::Const(c) if matches!(c.0, svod_ir::ConstValue::Float(f) if (f + 2.0).abs() < 1e-9))
        });
        assert!(has_neg2, "{arch:?}: the −2·cross scale is a Mul by the const −2.0");
        assert!(
            topo.iter().any(|u| matches!(u.op(), Op::Binary(svod_ir::BinaryOp::Add, ..))),
            "{arch:?}: the + c_sq add"
        );
    }
}

// =============================================================================
// Hardware-gated end-to-end score on gfx942 / gfx1151.
// =============================================================================

/// Whether the env-selected device is a supported AMD GPU (with the AMD-LLVM
/// toolchain) — else the `#[ignore]`d test self-skips instead of erroring on CPU.
fn device_supported() -> bool {
    let spec = svod_tensor::Tensor::empty(&[1], DType::Float32).device();
    crate::target::check_target(&spec, KNN_SUPPORTED_ARCHS).is_ok()
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib knn::test_knn_score_amd -- --ignored --nocapture`.
///
/// Random `x[query, d]`, `c[corpus, d]` (bf16); `c_sq[m] = Σ_d c[m,d]²` computed
/// host-side in f32 and replicated along the query axis. The kernel score is
/// compared to the generic-Tensor reference `c_sq[:,None] − 2·(c.f32 @ x.f32ᵀ)`,
/// oriented to the kernel's `score[m, n]` (corpus = row, query = col) tile. The
/// tolerance scales with √D like the matmul proptest (the cross term reduces over
/// D in bf16; `c_sq` is exact f32).
#[test]
#[ignore]
fn test_knn_score_amd() {
    use svod_tensor::Tensor;
    use svod_tensor::testing::allclose_f32;

    if !device_supported() {
        eprintln!("skip test_knn_score_amd: no supported AMD GPU / toolchain");
        return;
    }
    let dev = Tensor::rand(&[16, 16]).expect("probe").device();
    let arch = crate::target::resolve_arch(&dev).expect("resolve arch");
    let w = ArchCaps::for_arch(arch).wave_size as i64;

    for (corpus, query, d) in [(16usize, 16usize, 16usize), (32, 32, 32), (32, 16, 48)] {
        // Realized bf16 inputs so the kernel and the reference see identical rounding.
        let mut x = Tensor::randn(&[query, d]).expect("randn x").cast(DType::BFloat16).expect("x→bf16");
        let mut c = Tensor::randn(&[corpus, d]).expect("randn c").cast(DType::BFloat16).expect("c→bf16");
        x.realize().expect("realize x");
        c.realize().expect("realize c");

        // c_sq[m] = Σ_d c[m,d]² in f32, replicated to [corpus, query] (each (m,n) → c_sq[m]).
        let cf = c.cast(DType::Float32).expect("c→f32");
        let mut c_sq_rep = cf
            .try_mul(&cf)
            .expect("c²")
            .sum_with()
            .axes(1isize)
            .keepdim(true)
            .call()
            .expect("Σ_d c²")
            .try_expand([corpus, query])
            .expect("replicate along query");
        c_sq_rep.realize().expect("realize c_sq_rep");

        let mut score = Tensor::empty(&[1, 1, corpus, query], DType::Float32);
        crate::run_kernel("knn_score", [1, 1, 1], w, &mut [&mut score], &[&x, &c, &c_sq_rep], |ker| {
            build_knn_score(ker, corpus, query, d);
            ker.finish(1)
        })
        .expect("knn_score launch");
        let got = score.as_vec::<f32>().expect("read score");

        // Reference: c_sq[m] − 2·⟨c[m], x[n]⟩, oriented score[m, n] (corpus row, query col).
        let xf = x.cast(DType::Float32).expect("x→f32");
        let cross = cf.matmul(&xf.try_permute(&[1, 0]).expect("xᵀ")).expect("c @ xᵀ"); // [corpus, query]
        let two = Tensor::from_slice([2.0f32]);
        let mut refb = c_sq_rep.try_sub(&cross.try_mul(&two).expect("2·cross")).expect("c_sq − 2·cross");
        refb.realize().expect("realize ref");
        let exp = refb.as_vec::<f32>().expect("read ref");

        let (atol, rtol) = (0.02 * (d as f32).sqrt(), 2e-2);
        let r = allclose_f32(&got, &exp, atol, rtol);
        let max_abs = got.iter().zip(&exp).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
        println!("knn_score corpus={corpus} query={query} d={d}: max abs error = {max_abs:e} on {arch:?}");
        assert!(r.ok, "knn_score corpus={corpus} query={query} d={d} on {arch:?}: {}", r.message);
    }
}

// =============================================================================
// Stage 2 — running top-K (argmin-insert).
// =============================================================================

/// Placeholder buffers for a GPU-free topk build: `idx` (i32), `val` (f32), then
/// `x`/`c` (bf16), `c_sq_rep` (f32), in ABI order.
fn topk_bufs(corpus: usize, query: usize, d: usize, k: usize) -> Vec<Arc<UOp>> {
    vec![
        UOp::new_buffer(DeviceSpec::Cpu, query * k, DType::Int32),
        UOp::new_buffer(DeviceSpec::Cpu, query * k, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, query * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, corpus * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, corpus * query, DType::Float32),
    ]
}

/// Build the KNN topk SINK for `(corpus, query, d, k)` on `caps` (GPU-free).
fn topk_sink(corpus: usize, query: usize, d: usize, k: usize, caps: ArchCaps) -> Arc<UOp> {
    let ker = Kernel::new("knn_topk", [1, 1, 1], caps.wave_size as i64, topk_bufs(corpus, query, d, k), caps);
    build_knn_topk(&ker, corpus, query, d, k);
    ker.finish(2)
}

/// The topk kernel's graph carries the full argmin-insert machinery on BOTH archs:
/// the score WMMA, the index-carrying `row_arg_reduce` `ds_bpermute` `Op::Custom`
/// gathers (two reduces per insert step — a corpus-min and a K-slot-max — each
/// riding the arch's `reduce_tree`), the `Op::Ternary` evict/mask `where`s, and the
/// two `[query, k]` output stores. Built rolled (the corpus loop).
#[test]
fn test_knn_topk_graph_shape() {
    for caps in [ArchCaps::GFX942, ArchCaps::for_amd(AmdArch::Gfx1151)] {
        let arch = caps.arch;
        let (corpus, query, d, k) = (32usize, 16usize, 16usize, 4usize);
        let topo = topk_sink(corpus, query, d, k, caps).toposort();

        assert!(topo.iter().any(|u| matches!(u.op(), Op::Wmma(..))), "{arch:?}: score WMMA");

        // The arg-reduce cross-lane gathers (value + index each ride a ds_bpermute):
        // many across the k insert steps × 2 reduces × reduce_tree length.
        let customs = topo.iter().filter(|u| matches!(u.op(), Op::Custom(..))).count();
        assert!(customs >= 4 * caps.reduce_tree().len(), "{arch:?}: arg_reduce ds_bpermute Op::Customs, got {customs}");

        // The evict/mask conditional rewrites are `where` (Ternary) selects.
        let ternaries = topo.iter().filter(|u| matches!(u.op(), Op::Ternary(..))).count();
        assert!(ternaries >= k, "{arch:?}: evict/remove Ternary wheres, got {ternaries}");

        // The do_insert/tie predicates: Lt and Eq compares.
        assert!(
            topo.iter().any(|u| matches!(u.op(), Op::Binary(svod_ir::BinaryOp::Lt, ..))),
            "{arch:?}: do_insert Lt compare"
        );
        assert!(
            topo.iter().any(|u| matches!(u.op(), Op::Binary(svod_ir::BinaryOp::Eq, ..))),
            "{arch:?}: K-slot Eq compare"
        );

        // Two outputs: a Store into the i32 idx Param (slot 0) and the f32 val (slot 1).
        let stores_to = |slot: usize| {
            topo.iter().any(|u| {
                let Op::Store(..) = u.op() else { return false };
                u.toposort().iter().any(|s| matches!(s.op(), Op::Param(ops::Param { arg, .. }) if arg.slot == slot))
            })
        };
        assert!(stores_to(0), "{arch:?}: store into the idx output (Param 0)");
        assert!(stores_to(1), "{arch:?}: store into the val output (Param 1)");
    }
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib knn::test_knn_topk_amd -- --ignored --nocapture`.
///
/// Random `x[query, d]`, `c[corpus, d]` (bf16); `c_sq[m] = Σ_d c[m,d]²` in f32
/// replicated along query. The kernel emits the UNSORTED K nearest corpus indices
/// per query; the reference is the x²-free score `c_sq[m] − 2·⟨c[m],x[n]⟩` (as in
/// [`test_knn_score_amd`]), permuted to `[query, corpus]` and `topk(k, largest =
/// false)`. Both index lists are sorted per query and compared as sets (the kernel
/// is unsorted; ties → smaller corpus index, which both honor). Values compared at
/// √D-scaled tolerance. Covers a forced symmetric tie, a ragged corpus, `k = 1`,
/// and `k` near 16.
#[test]
#[ignore]
fn test_knn_topk_amd() {
    use svod_tensor::Tensor;
    use svod_tensor::testing::allclose_f32;

    if !device_supported() {
        eprintln!("skip test_knn_topk_amd: no supported AMD GPU / toolchain");
        return;
    }
    let dev = Tensor::rand(&[16, 16]).expect("probe").device();
    let arch = crate::target::resolve_arch(&dev).expect("resolve arch");
    let w = ArchCaps::for_arch(arch).wave_size as i64;

    // (corpus, query, d, k, tie): a square/ragged sweep over k. `tie` forces two
    // corpus rows equidistant to query 0 (a duplicated corpus row) so the smaller
    // corpus index must be kept.
    let cases: &[(usize, usize, usize, usize, bool)] = &[
        (32, 16, 16, 1, false),
        (32, 16, 16, 4, false),
        (32, 16, 32, 8, false),
        (48, 16, 16, 16, false),
        (40, 16, 16, 5, false), // ragged: 40 % 16 != 0
        (32, 16, 16, 1, true),  // forced tie on query 0 (k=1: the tied pair {0,1} competes
                                // for the single slot → the smaller index 0 must win)
    ];

    for &(corpus, query, d, k, tie) in cases {
        let mut x = Tensor::randn(&[query, d]).expect("randn x").cast(DType::BFloat16).expect("x→bf16");
        let mut c = Tensor::randn(&[corpus, d]).expect("randn c").cast(DType::BFloat16).expect("c→bf16");
        x.realize().expect("realize x");
        c.realize().expect("realize c");

        // Forced tie: make corpus row 1 a duplicate of corpus row 0 (so rows 0 and 1
        // are equidistant to every query → the smaller index, 0, must be kept), AND
        // point query 0 AT that row so the tied pair is query 0's nearest — otherwise a
        // random query 0 need not rank the pair in its top-k and the tie is untested.
        if tie {
            let row0 = c.try_shrink([(0, 1), (0, d as isize)]).expect("row0");
            let rest = c.try_shrink([(2, corpus as isize), (0, d as isize)]).expect("rest");
            c = Tensor::cat(&[&row0, &row0, &rest], 0).expect("dup row0 into row1");
            c.realize().expect("realize tie c");
            // x[0] = c[0]: query 0 sits on the duplicated rows (distance 0 to both 0 and 1).
            let q0 = c.try_shrink([(0, 1), (0, d as isize)]).expect("tie q0=c[0]");
            let qrest = x.try_shrink([(1, query as isize), (0, d as isize)]).expect("tie qrest");
            x = Tensor::cat(&[&q0, &qrest], 0).expect("set x[0]=c[0]");
            x.realize().expect("realize tie x");
        }

        // c_sq[m] = Σ_d c[m,d]² in f32, replicated to [corpus, query].
        let cf = c.cast(DType::Float32).expect("c→f32");
        let mut c_sq_rep = cf
            .try_mul(&cf)
            .expect("c²")
            .sum_with()
            .axes(1isize)
            .keepdim(true)
            .call()
            .expect("Σ_d c²")
            .try_expand([corpus, query])
            .expect("replicate along query");
        c_sq_rep.realize().expect("realize c_sq_rep");

        let mut idx_out = Tensor::empty(&[1, 1, query, k], DType::Int32);
        let mut val_out = Tensor::empty(&[1, 1, query, k], DType::Float32);
        crate::run_kernel("knn_topk", [1, 1, 1], w, &mut [&mut idx_out, &mut val_out], &[&x, &c, &c_sq_rep], |ker| {
            build_knn_topk(ker, corpus, query, d, k);
            ker.finish(2)
        })
        .expect("knn_topk launch");
        let got_idx = idx_out.as_vec::<i32>().expect("read idx");
        let got_val = val_out.as_vec::<f32>().expect("read val");

        // Reference: x²-free score [corpus, query] → [query, corpus] → topk smallest.
        let xf = x.cast(DType::Float32).expect("x→f32");
        let cross = cf.matmul(&xf.try_permute(&[1, 0]).expect("xᵀ")).expect("c @ xᵀ");
        let two = Tensor::from_slice([2.0f32]);
        let score = c_sq_rep.try_sub(&cross.try_mul(&two).expect("2·cross")).expect("score [corpus, query]");
        let score_qc = score.try_permute(&[1, 0]).expect("→[query, corpus]");
        let (mut ref_val, mut ref_idx) = score_qc.topk(k, -1, false).expect("ref topk");
        ref_val.realize().expect("realize ref_val");
        ref_idx.realize().expect("realize ref_idx");
        let exp_idx = ref_idx.as_vec::<i32>().expect("read ref idx");
        let exp_val = ref_val.as_vec::<f32>().expect("read ref val");

        // Compare per query as SETS (the kernel is unsorted): sort both index lists.
        let atol = 0.02 * (d as f32).sqrt();
        let mut ok = true;
        for q in 0..query {
            let mut g: Vec<i32> = got_idx[q * k..(q + 1) * k].to_vec();
            let mut e: Vec<i32> = exp_idx[q * k..(q + 1) * k].to_vec();
            g.sort_unstable();
            e.sort_unstable();
            if g != e {
                ok = false;
                eprintln!("query {q}: kernel idx {g:?} != ref idx {e:?} (corpus={corpus} k={k} tie={tie})");
            }
            // Values: the kept scores must match the reference's K scores as a set.
            let mut gv: Vec<f32> = got_val[q * k..(q + 1) * k].to_vec();
            let mut ev: Vec<f32> = exp_val[q * k..(q + 1) * k].to_vec();
            gv.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let r = allclose_f32(&gv, &ev, atol, 2e-2);
            if !r.ok {
                ok = false;
                eprintln!("query {q}: value mismatch {} (corpus={corpus} k={k})", r.message);
            }
        }
        // The forced tie (k=1, query 0 = corpus row 0): rows 0 and 1 are identical, so
        // both are at distance 0 — the nearest. With a single slot they tie for it, and
        // the smaller corpus index (0) must be kept and the duplicate (1) excluded
        // (`Tensor::topk`/`argmin`'s smaller-index convention, which `arg_fold` honors).
        if tie {
            let q0: Vec<i32> = got_idx[0..k].to_vec();
            assert!(q0.contains(&0), "tie: query 0 must keep the smaller corpus index 0, got {q0:?}");
            assert!(!q0.contains(&1), "tie: query 0 must NOT keep the duplicate index 1, got {q0:?}");
        }
        assert!(ok, "knn_topk corpus={corpus} query={query} d={d} k={k} tie={tie} on {arch:?}");
        println!("knn_topk corpus={corpus} query={query} d={d} k={k} tie={tie}: OK on {arch:?}");
    }
}

// =============================================================================
// Stage 3 — the public `svod_tk::knn` entry (prep + launch + generic-graph tail).
// =============================================================================

/// The malformed-request `Err` paths of the public [`crate::knn`] are checked BEFORE
/// arch resolution, so they hold on ANY device (no GPU required): a `D` mismatch
/// between the query and the corpus, `k > M`, and `k` outside the kernel's `1..=16`.
#[test]
fn test_knn_err_paths() {
    use svod_tensor::Tensor;

    // D mismatch: x is [4, 8], c is [16, 6] → query/corpus dims disagree.
    let x = Tensor::randn(&[4, 8]).expect("x");
    let c_baddim = Tensor::randn(&[16, 6]).expect("c bad-d");
    crate::knn(&x, &c_baddim, 2).err().expect("D mismatch must error, not None/panic");

    // k > M: only 3 corpus rows but k = 4.
    let c_small = Tensor::randn(&[3, 8]).expect("c small");
    crate::knn(&x, &c_small, 4).err().expect("k > M must error");

    // k > 16 (and k == 0) are outside the kernel's 1..=16.
    let c = Tensor::randn(&[64, 8]).expect("c");
    crate::knn(&x, &c, 17).err().expect("k > 16 must error");
    crate::knn(&x, &c, 0).err().expect("k == 0 must error");

    // Non-rank-2 (rank-1 query) errors via `concrete_dims`, also device-independent.
    let x1 = Tensor::randn(&[8]).expect("x rank-1");
    crate::knn(&x1, &c, 2).err().expect("rank-1 query must error");
}

/// Device-gated **graph-shape** check (builds the lazy `(dists, idxs)` but does NOT
/// realize — no dispatch): the `idxs` graph carries the kernel's `Op::Call`
/// custom-kernel node AND the generic-graph tail (the sort's `Max`/`Min` compares +
/// the gather's `Eq` select), and the outcome is `Ok(Some)`. Self-skips off an AMD GPU
/// (the public entry returns `Ok(None)` there — the kernel can't be built without a
/// resolved arch). Covers a ragged-D / multi-query-block shape so the tail's
/// pad/slice/reshape are exercised in the graph.
#[test]
fn test_knn_graph_shape() {
    use svod_tensor::Tensor;

    if !device_supported() {
        eprintln!("skip test_knn_graph_shape: no supported AMD GPU / toolchain");
        return;
    }
    let x = Tensor::randn(&[40, 20]).expect("x [40,20]"); // N=40 (multi-block), D=20 (ragged)
    let c = Tensor::randn(&[100, 20]).expect("c [100,20]");
    let (dists, idxs) = crate::knn(&x, &c, 5).expect("knn builds").expect("Ok(Some) on a supported device");

    let topo = idxs.uop().toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Call(..))), "idxs graph carries the kernel Op::Call node");
    // The bitonic sort folds the K via Max/Min compares; the gather selects via Eq.
    assert!(
        topo.iter().any(|u| matches!(u.op(), Op::Binary(svod_ir::BinaryOp::Max, ..))),
        "tail sort emits a Max compare"
    );
    assert!(
        topo.iter().any(|u| matches!(u.op(), Op::Binary(svod_ir::BinaryOp::Eq, ..))),
        "tail gather emits an Eq select"
    );
    // The exact-distance tail (dists) carries the same kernel Call plus its own sub/sum.
    let dtopo = dists.uop().toposort();
    assert!(dtopo.iter().any(|u| matches!(u.op(), Op::Call(..))), "dists graph carries the kernel Op::Call node");
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib knn::test_knn_amd -- --ignored --nocapture`.
///
/// End-to-end public [`crate::knn`] vs a generic-Tensor brute-force reference
/// (`knn_torch_naive`): the FULL squared-L2 `‖x‖² + ‖c‖² − 2·x@cᵀ` (clamped ≥ 0),
/// `.topk(k, dim=1, largest=false)`. Indices are compared as **sorted sets per query**
/// (ties → smaller index, which both `topk` and the kernel honor); distances within a
/// √D-scaled bf16 tolerance (the kernel's cross term is bf16; the returned ‖·‖² is the
/// exact-f32 re-gather). Sweeps: `N ≤ 16` (single block) and `N > 16` (N=40,
/// multi-block + ragged query), `D` not a multiple of 16 (D=20, zero-pad), a ragged
/// corpus (`M % 16 != 0`), `k = 1`, and a forced-tie corpus.
#[test]
#[ignore]
fn test_knn_amd() {
    use svod_tensor::Tensor;
    use svod_tensor::testing::allclose_f32;

    if !device_supported() {
        eprintln!("skip test_knn_amd: no supported AMD GPU / toolchain");
        return;
    }
    let dev = Tensor::rand(&[16, 16]).expect("probe").device();
    let arch = crate::target::resolve_arch(&dev).expect("resolve arch");

    // (n, m, d, k, tie): single-block, multi-block, ragged-D, ragged-corpus, k=1, tie.
    let cases: &[(usize, usize, usize, usize, bool)] = &[
        (12, 32, 16, 4, false),  // N ≤ 16 single block
        (40, 100, 16, 5, false), // N > 16 multi-block + ragged query (40 % 16 != 0)
        (16, 64, 20, 4, false),  // D = 20 ragged-D (zero-pad)
        (16, 50, 16, 3, false),  // M = 50 ragged corpus (50 % 16 != 0)
        (32, 48, 24, 1, false),  // k = 1, ragged-D, multi-block
        (16, 32, 16, 1, true),   // forced tie: smaller corpus index must win
    ];

    for &(n, m, d, k, tie) in cases {
        // Deterministic per-case inputs so the sweep is reproducible (the bf16-score
        // kernel vs the f32 oracle is sensitive to k-boundary near-ties; a fixed seed
        // keeps the run stable, and the tie-tolerant index check below admits the
        // legitimate same-distance swaps that do arise).
        svod_tensor::rand::manual_seed(0x4B4E_4E00 ^ (n * 131 + m * 17 + d * 7 + k) as u64);
        let mut x = Tensor::randn(&[n, d]).expect("randn x");
        let mut c = Tensor::randn(&[m, d]).expect("randn c");
        x.realize().expect("realize x");
        c.realize().expect("realize c");

        // Forced tie: duplicate corpus row 0 into row 1 (both equidistant to any query),
        // and put query 0 ON that row so the tied pair is its nearest — the smaller
        // corpus index (0) must be kept, the duplicate (1) excluded.
        if tie {
            let row0 = c.try_shrink([(0, 1), (0, d as isize)]).expect("row0");
            let rest = c.try_shrink([(2, m as isize), (0, d as isize)]).expect("rest");
            c = Tensor::cat(&[&row0, &row0, &rest], 0).expect("dup row0 into row1");
            c.realize().expect("realize tie c");
            let q0 = c.try_shrink([(0, 1), (0, d as isize)]).expect("tie q0=c[0]");
            let qrest = x.try_shrink([(1, n as isize), (0, d as isize)]).expect("tie qrest");
            x = Tensor::cat(&[&q0, &qrest], 0).expect("set x[0]=c[0]");
            x.realize().expect("realize tie x");
        }

        let (mut dists, mut idxs) = crate::knn(&x, &c, k).expect("knn builds").expect("Ok(Some) on AMD");
        dists.realize().expect("realize dists");
        idxs.realize().expect("realize idxs");
        let got_dist = dists.as_vec::<f32>().expect("read dists");
        let got_idx = idxs.as_vec::<i32>().expect("read idxs");

        // Brute-force reference: full ‖x−c‖² = ‖x‖² + ‖c‖² − 2·x@cᵀ (clamp ≥ 0), then
        // topk smallest. `rfull` is the [N, M] true distance matrix (for tie resolution).
        let (rdist, ridx, rfull) = knn_torch_naive(&x, &c, k);

        let atol = 0.02 * (d as f32).sqrt();
        let mut ok = true;
        for q in 0..n {
            let g: Vec<i32> = got_idx[q * k..(q + 1) * k].to_vec();
            let e: Vec<i32> = ridx[q * k..(q + 1) * k].to_vec();
            // The kernel scores the corpus in bf16, the oracle in f32; at the k-boundary
            // two near-equidistant corpus rows can rank-swap (a legitimate tie the two
            // precisions break differently). So compare by the TRUE (f32) distances, not
            // the raw index set: the multiset of the kernel-chosen indices' true distances
            // must match the oracle's top-k true distances (ascending), within tolerance.
            // This rejects a genuinely-wrong neighbor but admits a same-distance swap.
            let true_d = |idx: &[i32]| -> Vec<f32> {
                let mut v: Vec<f32> = idx.iter().map(|&i| rfull[q * m + i as usize]).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v
            };
            let (gd, ed) = (true_d(&g), true_d(&e));
            let r_idx = allclose_f32(&gd, &ed, atol, 2e-2);
            // The kernel's chosen indices must be distinct + in range (no slot leak).
            let mut gs = g.clone();
            gs.sort_unstable();
            gs.dedup();
            let distinct = gs.len() == k && g.iter().all(|&i| (0..m as i32).contains(&i));
            if !r_idx.ok || !distinct {
                ok = false;
                eprintln!(
                    "query {q}: kernel idx {g:?} (true d {gd:?}) != ref idx {e:?} (true d {ed:?}) \
                     distinct={distinct} (n={n} m={m} d={d} k={k} tie={tie})"
                );
            }
            // The RETURNED dists are the kernel's exact f32 re-gather, sorted ascending —
            // they must equal the oracle's k smallest true distances element-wise.
            let r = allclose_f32(&got_dist[q * k..(q + 1) * k], &rdist[q * k..(q + 1) * k], atol, 2e-2);
            if !r.ok {
                ok = false;
                eprintln!("query {q}: returned-dist mismatch {} (n={n} m={m} d={d} k={k})", r.message);
            }
        }
        if tie {
            let q0: Vec<i32> = got_idx[0..k].to_vec();
            assert!(q0.contains(&0), "tie: query 0 must keep the smaller corpus index 0, got {q0:?}");
            assert!(!q0.contains(&1), "tie: query 0 must NOT keep the duplicate index 1, got {q0:?}");
        }
        let max_dist_err = got_dist.iter().zip(&rdist).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
        assert!(ok, "knn n={n} m={m} d={d} k={k} tie={tie} on {arch:?}");
        println!("knn n={n} m={m} d={d} k={k} tie={tie}: OK on {arch:?}, max dist err = {max_dist_err:e}");
    }
}

/// Generic-Tensor brute-force KNN reference: full squared-L2 distance
/// `‖x[n]‖² + ‖c[m]‖² − 2·⟨x[n],c[m]⟩` (clamped ≥ 0 for the bf16-free f32 path), then
/// `topk(k, dim=1, largest=false)`. Returns `(dists [N·k], idxs [N·k], full [N·M])` —
/// the top-K distances/indices (ascending, ties → smaller index) plus the full true
/// distance matrix (so the caller can resolve k-boundary near-ties against the exact
/// distances). Computed entirely in f32 (no kernel) — the independent oracle.
fn knn_torch_naive(x: &svod_tensor::Tensor, c: &svod_tensor::Tensor, k: usize) -> (Vec<f32>, Vec<i32>, Vec<f32>) {
    use svod_tensor::Tensor;
    let xf = x.cast(DType::Float32).expect("x→f32");
    let cf = c.cast(DType::Float32).expect("c→f32");
    let x_sq = xf.try_mul(&xf).expect("x²").sum_with().axes(1isize).keepdim(true).call().expect("‖x‖²"); // [N,1]
    let c_sq = cf.try_mul(&cf).expect("c²").sum_with().axes(1isize).keepdim(true).call().expect("‖c‖²"); // [M,1]
    let cross = xf.matmul(&cf.try_permute(&[1, 0]).expect("cᵀ")).expect("x@cᵀ"); // [N,M]
    let two = Tensor::from_slice([2.0f32]);
    let c_sq_row = c_sq.try_permute(&[1, 0]).expect("‖c‖²ᵀ"); // [1,M]
    let mut dist = x_sq
        .try_add(&c_sq_row)
        .expect("‖x‖²+‖c‖²")
        .try_sub(&cross.try_mul(&two).expect("2·cross"))
        .expect("full ‖x−c‖²")
        .relu()
        .expect("clamp ≥ 0"); // [N,M], numerically ≥ 0
    let (mut rd, mut ri) = dist.topk(k, -1, false).expect("ref topk");
    rd.realize().expect("realize ref dist");
    ri.realize().expect("realize ref idx");
    dist.realize().expect("realize ref full dist");
    (
        rd.as_vec::<f32>().expect("ref dist"),
        ri.as_vec::<i32>().expect("ref idx"),
        dist.as_vec::<f32>().expect("ref full dist"),
    )
}
