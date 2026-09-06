//! Tests for the k-means assignment kernel ([`crate::kernels::kmeans`]):
//! a GPU-free graph-shape check (both archs) of the score WMMA + running-argmin
//! fold machinery, the malformed-request `Err` paths, and a hardware-gated
//! end-to-end check vs a generic-Tensor brute-force reference.

use std::sync::Arc;

use svod_dtype::{AmdArch, DType, DeviceSpec};
use svod_ir::{Op, UOp};

use crate::kernels::kmeans::{KMEANS_SUPPORTED_ARCHS, build_kmeans_assign};
use crate::{ArchCaps, Kernel};
use svod_ir::ops;

/// Placeholder buffers for a GPU-free build: `ids` (i32), `dist` (f32), then
/// `x`/`c` (bf16), `c_sq_rep` (f32), in ABI order.
fn kmeans_bufs(n: usize, k: usize, d: usize) -> Vec<Arc<UOp>> {
    vec![
        UOp::new_buffer(DeviceSpec::Cpu, n, DType::Int32),
        UOp::new_buffer(DeviceSpec::Cpu, n, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, n * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, k * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, k * 16, DType::Float32),
    ]
}

/// Build the kmeans assignment SINK for `(n, k, d)` on `caps` (GPU-free).
fn kmeans_sink(n: usize, k: usize, d: usize, caps: ArchCaps) -> Arc<UOp> {
    let ker = Kernel::new("kmeans_assign", [1, 1, 1], caps.wave_size as i64, kmeans_bufs(n, k, d), caps);
    build_kmeans_assign(&ker, n, k, d);
    ker.finish(2)
}

/// The assignment kernel's graph carries: a score WMMA, the index-carrying
/// `row_arg_reduce` `ds_bpermute` `Op::Custom` gathers (two reduces per tile —
/// a tile-min over centroids + a running-best extraction — each riding the
/// arch's `reduce_tree`), the `Op::Ternary` slot-0 update `where`s, and the two
/// `[N, 1]` output stores. Holds on both wave64 (gfx942) and wave32 (gfx1151).
#[test]
fn test_kmeans_assign_graph_shape() {
    for caps in [ArchCaps::GFX942, ArchCaps::for_amd(AmdArch::Gfx1151)] {
        let arch = caps.arch;
        let (n, k, d) = (32usize, 48usize, 32usize);
        let topo = kmeans_sink(n, k, d, caps).toposort();

        // The score term emits a WMMA.
        assert!(topo.iter().any(|u| matches!(u.op(), Op::Wmma(..))), "{arch:?}: score WMMA");

        // The arg-reduce cross-lane gathers (value + index each ride a ds_bpermute):
        // two reduces per centroid tile (tile-min + running-best) × reduce_tree length.
        let customs = topo.iter().filter(|u| matches!(u.op(), Op::Custom(..))).count();
        assert!(customs >= 4 * caps.reduce_tree().len(), "{arch:?}: arg_reduce ds_bpermute Op::Customs, got {customs}");

        // The slot-0 update conditional rewrites are `where` (Ternary) selects.
        let ternaries = topo.iter().filter(|u| matches!(u.op(), Op::Ternary(..))).count();
        assert!(ternaries >= 2, "{arch:?}: slot-0 update Ternary wheres, got {ternaries}");

        // Two outputs: a Store into the i32 ids Param (slot 0) and the f32 dist (slot 1).
        let stores_to = |slot: usize| {
            topo.iter().any(|u| {
                let Op::Store(..) = u.op() else { return false };
                u.toposort().iter().any(|s| matches!(s.op(), Op::Param(ops::Param { arg, .. }) if arg.slot == slot))
            })
        };
        assert!(stores_to(0), "{arch:?}: store into the ids output (Param 0)");
        assert!(stores_to(1), "{arch:?}: store into the dist output (Param 1)");
    }
}

// =============================================================================
// Hardware-gated end-to-end on gfx942 / gfx1151.
// =============================================================================

/// Whether the env-selected device is a supported AMD GPU (with the AMD-LLVM
/// toolchain) — else the `#[ignore]`d test self-skips instead of erroring on CPU.
fn device_supported() -> bool {
    let spec = svod_tensor::Tensor::empty(&[1], DType::Float32).device();
    crate::target::check_target(&spec, KMEANS_SUPPORTED_ARCHS).is_ok()
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib kmeans::test_kmeans_assign_amd -- --ignored --nocapture`.
///
/// Random `x[N, D]` and `c[K, D]` (bf16); the kernel's assignment is compared
/// against a brute-force f32 reference (full `‖x[n] − c[k]‖²` argmin). Indices
/// compared as exact matches (kmeans top-1 is deterministic — no sorting
/// ambiguity); distances within a √D-scaled bf16 tolerance.
#[test]
fn test_kmeans_assign_amd() {
    use svod_tensor::Tensor;
    use svod_tensor::testing::allclose_f32;

    svod_schedule::testing::setup_test_tracing();

    if !device_supported() {
        eprintln!("skip test_kmeans_assign_amd: no supported AMD GPU / toolchain");
        return;
    }
    let dev = Tensor::rand(&[16, 16]).expect("probe").device();
    let arch = crate::target::resolve_arch(&dev).expect("resolve arch");

    // (N, K, D): small/large K, ragged-K, ragged-D.
    let cases: &[(usize, usize, usize)] = &[
        (32, 16, 32),  // K = BLK (single tile)
        (32, 48, 32),  // K > TM (multi-tile, ragged)
        (32, 32, 32),  // K = TM (exactly one tile)
        (16, 64, 16),  // N = BLK (single block), D = BLK
        (48, 40, 32),  // ragged N (48 % 16 != 0), ragged K (40 % 32 != 0)
        (64, 256, 48), // larger K, ragged D (48 % 16 == 0 ✓ but tests wider D)
    ];

    for &(n, k, d) in cases {
        svod_tensor::rand::manual_seed(0x4B4D_0000 ^ (n * 131 + k * 17 + d * 7) as u64);
        let mut x = Tensor::randn(&[n, d]).expect("randn x");
        let mut c = Tensor::randn(&[k, d]).expect("randn c");
        x.realize().expect("realize x");
        c.realize().expect("realize c");

        let (mut ids, mut dists) = crate::kmeans_assign(&x, &c).expect("kmeans builds").expect("Ok(Some) on AMD");
        ids.realize().expect("realize ids");
        dists.realize().expect("realize dists");
        let got_ids = ids.as_vec::<i32>().expect("read ids");
        let got_dist = dists.as_vec::<f32>().expect("read dists");

        // Brute-force reference: full ‖x−c‖² = ‖x‖² + ‖c‖² − 2·x@cᵀ (clamp ≥ 0),
        // argmin over K.
        let (ref_ids, ref_dist) = kmeans_naive(&x, &c);

        let atol = 0.02 * (d as f32).sqrt();
        let mut ok = true;
        for i in 0..n {
            // The kernel scores in bf16, the oracle in f32 — near-tied centroids
            // can legitimately swap. Compare by the TRUE f32 distances: the
            // kernel-chosen index's true distance must be within tolerance of
            // the oracle's min distance.
            let ref_min = ref_dist[i];
            let kernel_true = ref_dist_full(&x, &c, i, got_ids[i] as usize);
            let r = allclose_f32(&[kernel_true], &[ref_min], atol, 2e-2);
            if !r.ok {
                ok = false;
                eprintln!(
                    "point {i}: kernel idx {} (true d {kernel_true:e}) != ref idx {} (true d {ref_min:e}) (n={n} k={k} d={d})",
                    got_ids[i], ref_ids[i]
                );
            }
            // The returned distance must match.
            let r = allclose_f32(&[got_dist[i]], &[ref_min], atol, 2e-2);
            if !r.ok {
                ok = false;
                eprintln!("point {i}: returned dist {} != ref min {} (n={n} k={k} d={d})", got_dist[i], ref_min);
            }
        }
        assert!(ok, "kmeans_assign n={n} k={k} d={d} on {arch:?}");
        println!("kmeans_assign n={n} k={k} d={d}: OK on {arch:?}");
    }
}

/// Generic-Tensor brute-force k-means assignment reference: full squared-L2
/// `‖x[n]‖² + ‖c[k]‖² − 2·⟨x[n],c[k]⟩` (clamped ≥ 0), argmin over K. Returns
/// `(ids [N], min_dist [N])` — entirely in f32 (no kernel).
fn kmeans_naive(x: &svod_tensor::Tensor, c: &svod_tensor::Tensor) -> (Vec<i32>, Vec<f32>) {
    let xf = x.cast(DType::Float32).expect("x→f32");
    let cf = c.cast(DType::Float32).expect("c→f32");
    let x_dims = x.shape().expect("x shape");
    let n = x_dims[0].as_const().expect("N");
    let k = c.shape().expect("c shape")[0].as_const().expect("K");

    let x_sq = xf.try_mul(&xf).expect("x²").sum_with().axes(1isize).keepdim(true).call().expect("‖x‖²"); // [N,1]
    let c_sq = cf.try_mul(&cf).expect("c²").sum_with().axes(1isize).keepdim(true).call().expect("‖c‖²"); // [K,1]
    let cross = xf.matmul(&cf.try_permute(&[1, 0]).expect("cᵀ")).expect("x@cᵀ"); // [N,K]
    let two = svod_tensor::Tensor::from_slice([2.0f32]);
    let c_sq_row = c_sq.try_permute(&[1, 0]).expect("‖c‖²ᵀ"); // [1,K]
    let mut dist = x_sq
        .try_add(&c_sq_row)
        .expect("‖x‖²+‖c‖²")
        .try_sub(&cross.try_mul(&two).expect("2·cross"))
        .expect("full ‖x−c‖²")
        .relu()
        .expect("clamp ≥ 0"); // [N,K]
    dist.realize().expect("realize ref dist");
    let dist_v = dist.as_vec::<f32>().expect("dist vec");

    let mut ids = vec![-1i32; n];
    let mut min_d = vec![0.0f32; n];
    for i in 0..n {
        let mut best = f32::INFINITY;
        let mut best_j = 0;
        for j in 0..k {
            let d = dist_v[i * k + j];
            if d < best {
                best = d;
                best_j = j as i32;
            }
        }
        ids[i] = best_j;
        min_d[i] = best;
    }
    (ids, min_d)
}

/// The true squared-L2 distance `‖x[i] − c[j]‖²` in f32 (for tie-breaking).
fn ref_dist_full(x: &svod_tensor::Tensor, c: &svod_tensor::Tensor, i: usize, j: usize) -> f32 {
    let xf = x.cast(DType::Float32).expect("x→f32");
    let cf = c.cast(DType::Float32).expect("c→f32");
    let d = x.shape().expect("x shape")[1].as_const().expect("D");
    let xi = xf.try_shrink([(i as isize, (i + 1) as isize), (0, d as isize)]).expect("xi");
    let cj = cf.try_shrink([(j as isize, (j + 1) as isize), (0, d as isize)]).expect("cj");
    let diff = xi.try_sub(&cj).expect("diff");
    let mut dist = diff.try_mul(&diff).expect("diff²").sum_with().axes(1isize).call().expect("Σ diff²");
    dist.realize().expect("realize ref dist_full");
    dist.as_vec::<f32>().expect("dist vec")[0]
}

// =============================================================================
// The public entry — malformed-request `Err` paths + graph-shape check.
// =============================================================================

/// The malformed-request `Err` paths of [`crate::kmeans_assign`] are checked
/// BEFORE arch resolution, so they hold on ANY device (no GPU required): a `D`
/// mismatch between the points and the centroids, and non-rank-2 operands.
#[test]
fn test_kmeans_assign_err_paths() {
    use svod_tensor::Tensor;

    // D mismatch: x is [32, 64], c is [16, 32].
    let x = Tensor::randn(&[32, 64]).expect("x");
    let c_bad = Tensor::randn(&[16, 32]).expect("c bad-D");
    crate::kmeans_assign(&x, &c_bad).err().expect("D mismatch must error");

    // Non-rank-2 (rank-1 points).
    let x1 = Tensor::randn(&[64]).expect("x rank-1");
    let c = Tensor::randn(&[16, 64]).expect("c");
    crate::kmeans_assign(&x1, &c).err().expect("rank-1 points must error");

    // Non-rank-2 (rank-3 centroids).
    let c3 = Tensor::randn(&[1, 16, 64]).expect("c rank-3");
    crate::kmeans_assign(&x, &c3).err().expect("rank-3 centroids must error");
}

/// Device-gated **graph-shape** check (builds the lazy `(ids, dists)` but does
/// NOT realize — no dispatch): the graph carries the kernel's `Op::Call` node
/// and the tail's `‖x‖²` re-add (a `Mul` + `Sum`). Self-skips off an AMD GPU.
#[test]
fn test_kmeans_assign_public_graph_shape() {
    use svod_tensor::Tensor;

    if !device_supported() {
        eprintln!("skip test_kmeans_assign_public_graph_shape: no supported AMD GPU / toolchain");
        return;
    }
    let x = Tensor::randn(&[48, 32]).expect("x [48,32]"); // ragged N
    let c = Tensor::randn(&[64, 32]).expect("c [64,32]");
    let (ids, dists) = crate::kmeans_assign(&x, &c).expect("builds").expect("Ok(Some) on a supported device");

    let topo = ids.uop().toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Call(..))), "ids graph carries the kernel Op::Call");
    let dtopo = dists.uop().toposort();
    assert!(dtopo.iter().any(|u| matches!(u.op(), Op::Call(..))), "dists graph carries the kernel Op::Call");
}

// =============================================================================
// kmeans_update — generic-graph centroid update.
// =============================================================================

/// [`crate::kmeans_update`] against a brute-force f32 reference: one Lloyd
/// update step on random data. Holds on any device (pure generic-graph, no
/// tile kernel).
#[test]
fn test_kmeans_update_vs_reference() {
    use svod_tensor::Tensor;

    let (n, k, d) = (64usize, 8usize, 16usize);
    svod_tensor::rand::manual_seed(0x4B4D_0001);
    let x = Tensor::randn(&[n, d]).expect("randn x");
    let c = Tensor::randn(&[k, d]).expect("randn c");

    // Random assignments in [0, k).
    let ids_vec: Vec<i32> = (0..n).map(|i| (i % k) as i32).collect();
    let ids = Tensor::from_slice(&ids_vec).cast(DType::Int32).expect("ids");

    let (new_c, shift) = crate::kmeans_update(&x, &ids, &c).expect("kmeans_update");
    let mut nc = new_c;
    let mut sh = shift;
    nc.realize().expect("realize new_c");
    sh.realize().expect("realize shift");
    let got_c = nc.as_vec::<f32>().expect("new_c vec");
    let got_shift = sh.as_vec::<f32>().expect("shift vec");

    // Reference: per-cluster mean of assigned points, shift = ‖new − old‖.
    let mut xf_t = x.cast(DType::Float32).expect("x→f32");
    let mut cf_t = c.cast(DType::Float32).expect("c→f32");
    xf_t.realize().expect("realize xf");
    cf_t.realize().expect("realize cf");
    let xf = xf_t.as_vec::<f32>().expect("xf vec");
    let cf = cf_t.as_vec::<f32>().expect("cf vec");
    let mut ref_c = vec![0.0f32; k * d];
    let mut counts = vec![0usize; k];
    for i in 0..n {
        let cl = ids_vec[i] as usize;
        for j in 0..d {
            ref_c[cl * d + j] += xf[i * d + j];
        }
        counts[cl] += 1;
    }
    for cl in 0..k {
        if counts[cl] > 0 {
            for j in 0..d {
                ref_c[cl * d + j] /= counts[cl] as f32;
            }
        } else {
            // Empty cluster: reuse old centroid.
            for j in 0..d {
                ref_c[cl * d + j] = cf[cl * d + j];
            }
        }
    }
    let r = svod_tensor::testing::allclose_f32(&got_c, &ref_c, 1e-4, 1e-4);
    assert!(r.ok, "kmeans_update centroids: {}", r.message);

    // Check shift.
    let mut ref_shift = vec![0.0f32; k];
    for cl in 0..k {
        let mut s = 0.0f32;
        for j in 0..d {
            let diff = ref_c[cl * d + j] - cf[cl * d + j];
            s += diff * diff;
        }
        ref_shift[cl] = s.sqrt();
    }
    let r = svod_tensor::testing::allclose_f32(&got_shift, &ref_shift, 1e-4, 1e-4);
    assert!(r.ok, "kmeans_update shift: {}", r.message);
}

/// [`crate::kmeans_update`] `Err` paths: D mismatch, rank errors, N mismatch.
#[test]
fn test_kmeans_update_err_paths() {
    use svod_tensor::Tensor;

    let x = Tensor::randn(&[32, 64]).expect("x");
    let ids = Tensor::from_slice([0i32, 1, 2, 0]).cast(DType::Int32).expect("ids");
    let c_ok = Tensor::randn(&[4, 64]).expect("c ok");

    // D mismatch: c has D=32, x has D=64.
    let c_bad = Tensor::randn(&[4, 32]).expect("c bad-D");
    crate::kmeans_update(&x, &ids, &c_bad).err().expect("D mismatch must error");

    // N mismatch: ids has 4 elements, x has 32 rows.
    crate::kmeans_update(&x, &ids, &c_ok).err().expect("N mismatch must error");

    // Non-rank-1 ids.
    let ids2 = Tensor::from_slice([0i32, 1]).cast(DType::Int32).expect("ids rank-1");
    let ids2 = ids2.try_reshape([1isize, 2]).expect("ids 2D");
    let x_small = Tensor::randn(&[2, 64]).expect("x small");
    let c_small = Tensor::randn(&[4, 64]).expect("c small");
    crate::kmeans_update(&x_small, &ids2, &c_small).err().expect("rank-2 ids must error");
}

// ── generic-baseline phi-dominance regression guard ─────────────────────────
//
// The generic GEMM-argmin kmeans baseline (`x·cᵀ → ‖x‖²+‖c‖²−2·x·cᵀ → min over K`,
// identical to `benches/kmeans.rs`) fuses the matmul and the min-over-K into one
// kernel. With REALIZED bf16 inputs (as the bench produces) the optimizer applied a
// tensor core to the matmul and tiled its N-output axis — which is *also* the min's
// reduce axis K — into Warp/Local sub-axes, so the min reduced over ranges shared
// with the GEMM and one loop got closed by two ENDs → invalid LLVM IR ("instruction
// does not dominate all uses"). The TC heuristic now declines a matmul whose output
// feeds a downstream reduce. This guards the whole bench K-sweep without criterion.
//
// Run: `SVOD_DEVICE=AMD:0 cargo test -p svod-tk -- --ignored kmeans_generic_phi`

/// The generic GEMM-argmin kmeans baseline — identical to `benches/kmeans.rs`.
fn kmeans_generic_ref(xb: &svod_tensor::Tensor, cb: &svod_tensor::Tensor) -> svod_tensor::Tensor {
    let f32 = DType::Float32;
    let xf = xb.cast(f32.clone()).expect("x→f32");
    let cf = cb.cast(f32.clone()).expect("c→f32");
    let x_sq = xf.try_mul(&xf).expect("x²").sum_with().axes(1isize).keepdim(true).call().expect("Σx²");
    let c_sq = cf.try_mul(&cf).expect("c²").sum_with().axes(1isize).keepdim(true).call().expect("Σc²");
    let c_sq_row = c_sq.try_transpose(0, 1).expect("c_sq→[1,K]");
    let ct = cb.try_transpose(0, 1).expect("cᵀ");
    let cross = xb.matmul_with().other(&ct).dtype(f32).call().expect("x·cᵀ");
    let two_cross = cross.try_add(&cross).expect("2·cross");
    let dist = x_sq.try_add(&c_sq_row).expect("‖x‖²+‖c²").try_sub(&two_cross).expect("−2·cross");
    dist.min(1).expect("min over K")
}

#[test]
#[ignore = "requires SVOD_DEVICE=AMD:0 with gfx1151"]
fn kmeans_generic_phi_dominance_mre() {
    use svod_tensor::Tensor;
    let (n, d) = (2048usize, 64usize);
    // Sweep the bench's centroid counts. Inputs are REALIZED (as the bench
    // produces them) so the matmul + min-over-K fuse into one kernel — the regime
    // that exposed the tensor-core-tiled-reduce-axis phi-dominance.
    for k in [64usize, 256, 1024, 4096] {
        let mut xb = Tensor::randn(&[n, d]).expect("x").cast(DType::BFloat16).expect("x→bf16");
        let mut cb = Tensor::randn(&[k, d]).expect("c").cast(DType::BFloat16).expect("c→bf16");
        xb.realize().expect("realize xb");
        cb.realize().expect("realize cb");

        let mut result = kmeans_generic_ref(&xb, &cb);
        result.prepare().unwrap_or_else(|e| panic!("kmeans_generic prepare failed (K={k}): {e}"));
    }
}
