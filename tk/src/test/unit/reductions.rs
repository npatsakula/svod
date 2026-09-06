//! Cross-lane reductions (`row_reduce`) and an end-to-end softmax — a port of
//! tinygrad `test_tk.py::test_softmax`.
//!
//! The graph-shape checks run GPU-free; the softmax comparison is `#[ignore]`
//! and validates on gfx942 (lane-distributed, so the CPU backend can't run it).

use smallvec::smallvec;
use svod_dtype::{AmdArch, DType};
use svod_ir::{BinaryOp, Op};

use crate::arch::FragRole;
use crate::tile::RegTile;
use crate::tiles::{RT_16X16, TileLayout, VecLayout};
use crate::{ArchCaps, ArgDir, Kernel, MoveIdx};
use svod_ir::ops;

const ROW: TileLayout = TileLayout::Row;

const INV_LN2: f64 = std::f64::consts::LOG2_E; // 1 / ln(2) == log2(e)

/// Build the softmax-over-axis-3 SINK for a `block × n` row-softmax with a
/// `block × block` tile, mirroring tinygrad `test_softmax`, on the arch's
/// accumulator fragment and canonical LDS strip.
fn build_softmax(ker: &Kernel, n: usize, block: usize) {
    let warp = ker.warp();

    // out (b, f32), then in (a, f32).
    let b = ker.gl(&[1, 1, block, n], DType::Float32);
    let a = ker.gl(&[1, 1, block, n], DType::Float32);

    let max_vec = ker.acc_vec(block);
    let norm_vec = ker.acc_vec(block);
    let max_vec_last = ker.acc_vec(block);

    let mut max_vec = warp.neg_inf_rv(max_vec);
    let mut norm_vec = warp.zero_rv(norm_vec);
    let mut max_vec_last = max_vec_last;

    // Pass 1: running max + normalization accumulator over the column tiles.
    let tile_col = ker.range((n / block) as i64);
    {
        let a_smem = ker.shared((block, block), DType::Float32, TileLayout::Row);
        let a_reg = ker.acc((block, block), TileLayout::Row);
        let a_smem = warp.load(a_smem, a.clone(), MoveIdx::block((0, 0, 0, tile_col.clone()), 2));
        let a_reg = warp.load(a_reg, a_smem, MoveIdx::default());
        let a_reg = warp.mul_scalar(a_reg, INV_LN2);

        max_vec_last = warp.copy(max_vec_last.after(smallvec![tile_col.clone()]), &max_vec);
        let mv_in = max_vec.after(smallvec![max_vec_last.uop().clone()]);
        max_vec = warp.row_reduce(mv_in, &a_reg, |x, y| x.try_max(y).expect("row max"), f64::NEG_INFINITY);

        let a_reg = warp.sub_rv(a_reg, &max_vec);
        let a_reg = warp.exp2(a_reg);
        max_vec_last = warp.exp2(warp.sub(max_vec_last, &max_vec));
        norm_vec = warp.mul(norm_vec, &max_vec_last);
        norm_vec = warp.row_reduce(norm_vec, &a_reg, |x, y| x.try_add(y).expect("row add"), 0.0);
    }
    norm_vec = norm_vec.rewrap(ker.endrange(1));
    max_vec = max_vec.after(smallvec![norm_vec.uop().clone()]);

    // Pass 2: recompute the (scaled) exponentials and normalize.
    let tile_col = ker.range((n / block) as i64);
    {
        let a_smem = ker.shared((block, block), DType::Float32, TileLayout::Row);
        let a_reg = ker.acc((block, block), TileLayout::Row);
        let a_smem = warp.load(a_smem, a.clone(), MoveIdx::block((0, 0, 0, tile_col.clone()), 2));
        let a_reg = warp.load(a_reg, a_smem, MoveIdx::default());
        let a_reg = warp.mul_scalar(a_reg, INV_LN2);
        let a_reg = warp.sub_rv(a_reg, &max_vec);
        let a_reg = warp.exp2(a_reg);
        let a_reg = warp.div_rv(a_reg, &norm_vec);
        let _ = warp.store(b, a_reg, MoveIdx::block((0, 0, 0, tile_col), 2));
    }
}

/// A bare `row_reduce` folds the three sibling 16-lane slots with an in-register
/// `ds_bpermute` wave shuffle (`Op::Custom`) — no LDS scratch (`DefineLocal`),
/// no workgroup `Barrier`, and no WMMA.
#[test]
fn test_row_reduce_graph_shape() {
    let ker = Kernel::new("row_reduce_probe", [1, 1, 1], 64, vec![], crate::ArchCaps::GFX942);
    let warp = ker.warp();

    let src = ker.rt((32, 32), DType::Float32, TileLayout::Row, RT_16X16);
    let src = warp.zero(src);
    let vec = ker.rv(32, DType::Float32, VecLayout::Ortho, RT_16X16);
    let vec = warp.zero_rv(vec);
    let out = warp.row_reduce(vec, &src, |x, y| x.try_add(y).expect("add"), 0.0);

    let topo = out.uop().toposort();
    assert!(
        topo.iter().any(|u| matches!(u.op(), Op::Custom(..))),
        "row_reduce gathers sibling lanes with a ds_bpermute Op::Custom shuffle"
    );
    assert!(
        !topo
            .iter()
            .any(|u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local))),
        "the wave-shuffle reduce allocates no LDS scratch"
    );
    assert!(
        !topo.iter().any(|u| matches!(u.op(), Op::Barrier(..))),
        "the wave-shuffle reduce needs no workgroup barrier"
    );
    assert!(!topo.iter().any(|u| matches!(u.op(), Op::Wmma(..))), "a reduction has no WMMA");
}

/// `row_arg_reduce` threads an index alongside the value: each `reduce_tree` step
/// shuffles BOTH payloads (so the partner's index rides its own `ds_bpermute`,
/// never re-derived) → exactly `2 * reduce_tree().len()` `Op::Custom` gathers, plus
/// the `where`-select pair-fold (`Ternary` + `Lt`/`Eq` compares), and still no LDS,
/// no barrier, no WMMA. Holds on both wave64 (gfx942) and wave32 (gfx1151).
#[test]
fn test_row_arg_reduce_graph_shape() {
    let build = |caps: ArchCaps| {
        let ker = Kernel::new("argred", [1, 1, 1], caps.wave_size as i64, vec![], caps);
        let warp = ker.warp();
        let frag = ker.frag(FragRole::Accumulator);
        let src = warp.zero(ker.rt((16, 16), DType::Float32, ROW, frag));
        let val = warp.clear_rv(ker.rv(16, DType::Float32, VecLayout::Ortho, frag), f64::INFINITY);
        let idx = warp.clear_rv(ker.rv(16, DType::Int32, VecLayout::Ortho, frag), -1.0);
        // The index result transitively depends on the value path (the keep
        // predicate reads the value compares), so its toposort covers both.
        let (_, idx) = warp.row_arg_reduce(val, idx, &src, ArgDir::Min);
        idx.uop().toposort()
    };
    for caps in [ArchCaps::GFX942, ArchCaps::for_amd(AmdArch::Gfx1151)] {
        let want_customs = 2 * caps.reduce_tree().len();
        let topo = build(caps);
        let customs = topo.iter().filter(|u| matches!(u.op(), Op::Custom(..))).count();
        assert_eq!(customs, want_customs, "{:?}: value+index each ride a ds_bpermute per tree step", caps.arch);
        assert!(topo.iter().any(|u| matches!(u.op(), Op::Ternary(..))), "{:?}: where-select pair-fold", caps.arch);
        assert!(
            topo.iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Lt, ..))),
            "{:?}: strict/tie Lt compare",
            caps.arch
        );
        assert!(topo.iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Eq, ..))), "{:?}: tie Eq compare", caps.arch);
        assert!(
            !topo
                .iter()
                .any(|u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local))),
            "{:?}: no LDS scratch",
            caps.arch
        );
        assert!(!topo.iter().any(|u| matches!(u.op(), Op::Barrier(..))), "{:?}: no barrier", caps.arch);
        assert!(!topo.iter().any(|u| matches!(u.op(), Op::Wmma(..))), "{:?}: no WMMA", caps.arch);
    }
}

// =============================================================================
// Hardware-gated end-to-end softmax (gfx942 wave64, CUDA warp32).
// =============================================================================

/// The single-warp softmax runs on a device whose `Row` accumulator map folds
/// columns per lane row — CDNA (`ds_bpermute` sibling tree) and CUDA (`shfl.bfly`
/// quad, two row slots per lane); RDNA's even/odd accumulator folds rows, so it
/// skips. Returns the wave width for the launch block.
fn softmax_device() -> Option<i64> {
    super::row_fold_device().map(|caps| caps.wave_size as i64)
}

/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib reductions::test_softmax_gpu -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_softmax_gpu() {
    use svod_tensor::Tensor;

    let Some(w) = softmax_device() else {
        eprintln!("skip test_softmax_gpu: no CDNA/CUDA device");
        return;
    };
    let (n, block) = (64usize, 32usize);

    let a = Tensor::rand(&[1, 1, block, n]).expect("rand a");
    let mut a = a.cast(DType::Float32).expect("cast a");
    a.realize().expect("realize a");
    let mut out = Tensor::empty(&[1, 1, block, n], DType::Float32);

    crate::run_kernel("softmax", [1, 1, 1], w, &mut [&mut out], &[&a], |ker| {
        build_softmax(ker, n, block);
        ker.finish(1)
    })
    .expect("softmax launch");

    let got = out.as_vec::<f32>().expect("read out");

    let mut reference = a.softmax(3isize).expect("ref softmax");
    reference.realize().expect("realize reference");
    let expected = reference.as_vec::<f32>().expect("read reference");

    assert_eq!(got.len(), expected.len(), "length mismatch");
    let max_abs = got.iter().zip(&expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
    println!("softmax N={n} block={block}: max abs error = {max_abs:e}");
    assert!(max_abs < 1e-4, "max abs error {max_abs} exceeds f32 softmax tolerance 1e-4");
}

/// P1 isolation (`SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib reductions::test_softmax_unroll_gpu -- --ignored --nocapture`):
/// the **fully-unrolled** softmax (reduce_u + unrolled map/copy, no mma/db) must
/// match the reference — isolates the unrolled reduce/elementwise from the FA
/// double-buffer + mma context. Swept across block sizes so `outer_end` varies
/// (16 → 1 fragment, 32 → 2).
#[test]
#[ignore]
fn test_softmax_unroll_gpu() {
    use svod_tensor::Tensor;

    let Some(w) = softmax_device() else {
        eprintln!("skip test_softmax_unroll_gpu: no CDNA/CUDA device");
        return;
    };
    for (n, block) in [(64usize, 16usize), (64, 32)] {
        let a = Tensor::rand(&[1, 1, block, n]).expect("rand a");
        let mut a = a.cast(DType::Float32).expect("cast a");
        a.realize().expect("realize a");
        let mut out = Tensor::empty(&[1, 1, block, n], DType::Float32);

        crate::run_kernel("softmax_u", [1, 1, 1], w, &mut [&mut out], &[&a], |ker| {
            ker.set_unroll(true);
            build_softmax(ker, n, block);
            ker.finish(1)
        })
        .expect("softmax_u launch");

        let got = out.as_vec::<f32>().expect("read out");
        let mut reference = a.softmax(3isize).expect("ref softmax");
        reference.realize().expect("realize reference");
        let expected = reference.as_vec::<f32>().expect("read reference");

        let max_abs = got.iter().zip(&expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
        println!("softmax_u N={n} block={block}: max abs error = {max_abs:e}");
        assert!(max_abs < 1e-4, "softmax_u N={n} block={block}: max abs error {max_abs} exceeds 1e-4");
    }
}

/// `SVOD_DEVICE={AMD,CUDA}:0 cargo test -p svod-tk --lib reductions::test_row_argmin_gpu -- --ignored --nocapture`.
///
/// End-to-end argmin of a known 16×16 matrix into the role-selected accumulator
/// fragment (arch-portable: wave64 gfx942, wave32 gfx1151, warp32 CUDA — where the
/// quad butterfly completes the fold and each lane keeps two row slots). `row_arg_reduce`
/// reduces the fragment's `inner`-carrying folded axis — the matrix *column* on the
/// non-interleave gfx942 frag, the matrix *row* on the wave32 even/odd accumulator
/// (the caller arranges the tile, exactly like `row_reduce` in FA). To assert one
/// result on either layout the matrix is **symmetric**, so the per-row argmin equals
/// the per-column argmin, and we read the output **diagonal** `out[k][k]`, which is
/// `rv[k]` whether the result vector broadcasts along rows or columns. An involution
/// pairs `(2k, 2k+1)` as the −1.0 minima; an extra symmetric −1.0 at `(0,6)`/`(6,0)`
/// makes index 0 a **tie** (cols 1 and 6 → must resolve to 1).
#[test]
#[ignore]
fn test_row_argmin_gpu() {
    use svod_tensor::Tensor;

    let Some(caps) = super::fragment_device() else {
        eprintln!("skip test_row_argmin_gpu: no device with tk fragment layouts");
        return;
    };
    let w = caps.wave_size as i64;

    // Symmetric matrix: +1.0 except −1.0 at the involution pairs (2k, 2k+1) and a
    // tie pair (0, 6). Symmetry ⇒ per-row argmin == per-column argmin, so the
    // expectation is layout-independent.
    let mut m = vec![1.0f32; 256];
    let mut set = |i: usize, j: usize| {
        m[i * 16 + j] = -1.0;
        m[j * 16 + i] = -1.0;
    };
    for k in 0..8 {
        set(2 * k, 2 * k + 1);
    }
    set(0, 6); // tie: row/col 0 now has −1.0 at {1, 6} → argmin 1; row/col 6 at {0, 7} → 0
    let mut expect = [0i32; 16];
    for r in 0..16usize {
        let (mut best_v, mut best_j) = (f32::INFINITY, 0i32);
        for c in 0..16usize {
            if m[r * 16 + c] < best_v {
                best_v = m[r * 16 + c];
                best_j = c as i32;
            }
        }
        expect[r] = best_j;
    }

    let mut a = Tensor::from_slice(&m).try_reshape([1usize, 1, 16, 16]).expect("reshape a");
    a.realize().expect("realize a");
    let mut vout = Tensor::empty(&[1, 1, 16, 16], DType::Float32);
    let mut iout = Tensor::empty(&[1, 1, 16, 16], DType::Int32);

    crate::run_kernel("argmin", [1, 1, 1], w, &mut [&mut vout, &mut iout], &[&a], |ker| {
        let warp = ker.warp();
        let frag = ker.frag(FragRole::Accumulator);
        let vo = ker.gl(&[1, 1, 16, 16], DType::Float32);
        let io = ker.gl(&[1, 1, 16, 16], DType::Int32);
        let ain = ker.gl(&[1, 1, 16, 16], DType::Float32);

        let src = warp.load(ker.rt((16, 16), DType::Float32, ROW, frag), ain, MoveIdx::block((0, 0, 0, 0), 2));
        let val = warp.clear_rv(ker.rv(16, DType::Float32, VecLayout::Ortho, frag), f64::INFINITY);
        let idx = warp.clear_rv(ker.rv(16, DType::Int32, VecLayout::Ortho, frag), -1.0);
        let (val, idx) = warp.row_arg_reduce(val, idx, &src, ArgDir::Min);

        let vtile = warp.add_rv(warp.zero(ker.rt((16, 16), DType::Float32, ROW, frag)), &val);
        let itile = warp.add_rv(warp.zero(ker.rt((16, 16), DType::Int32, ROW, frag)), &idx);
        let _ = warp.store(vo, vtile, MoveIdx::block((0, 0, 0, 0), 2));
        let _ = warp.store(io, itile, MoveIdx::block((0, 0, 0, 0), 2));
        ker.finish(2)
    })
    .expect("argmin launch");

    let gv = vout.as_vec::<f32>().expect("read vout");
    let gi = iout.as_vec::<i32>().expect("read iout");
    // Read the diagonal: out[k][k] == rv[k] regardless of the RV broadcast axis.
    for k in 0..16usize {
        assert_eq!(gi[k * 16 + k], expect[k], "row/col {k}: argmin index (diagonal)");
        assert!((gv[k * 16 + k] + 1.0).abs() < 1e-6, "row/col {k}: argmin value −1.0, got {}", gv[k * 16 + k]);
    }
    println!("row_argmin: 16/16 correct on {:?} (tie at index 0 → 1)", caps.arch);
}
