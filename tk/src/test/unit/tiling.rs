//! The tile model against the tiles measurement actually picked.
//!
//! The model is not asked to name the winner — [`crate::tune`] measures that —
//! but the winner has to be among the few candidates it offers, or the search
//! can never reach it. These cases pin that against an RTX 3060 (sm_86) and the
//! convolution shapes YOLO26-x tunes on it, whose per-candidate times were
//! measured on the hardware.

use svod_dtype::DType;
use test_case::test_case;

use crate::kernels::conv::ConvGeom;
use crate::kernels::gemm::{GemmCfg, NT_128X64};
use crate::kernels::tiling::{TileBudget, TripCost};
use crate::target::WorkgroupLimits;

/// An RTX 3060 (GA106): 28 SMs, 48 KiB of shared memory a workgroup, 100 KiB an
/// SM, a 64 K register file an SM, `m16n8k16` on a 32-lane warp.
fn rtx_3060() -> TileBudget {
    TileBudget {
        wave_size: 32,
        limits: WorkgroupLimits {
            max_threads: 1024,
            shared_per_workgroup: 48 * 1024,
            shared_per_cu: 100 * 1024,
            registers_per_cu: Some(64 * 1024),
        },
        mma_edge: 16,
        blocks_wanted: 28,
        resident_floor: 2,
    }
}

/// `(block_m, block_n, k_step)` of each candidate, in rank order, under the
/// kernel's own tiling rule.
fn ranked(
    budget: &TileBudget,
    trip: TripCost,
    (m, n): (usize, usize),
    limit: usize,
    accept: impl Fn(&GemmCfg) -> bool,
) -> Vec<(usize, usize, usize)> {
    budget
        .ranked(&NT_128X64, DType::Float16.bytes(), trip, (m, n), limit, accept)
        .into_iter()
        .map(|cfg| (cfg.block_m, cfg.block_n, cfg.k_step))
        .collect()
}

/// One of YOLO26-x's 3x3 convolutions: `cin -> cout` over a `side x side`
/// input at `stride`, batch 1, as the model holds it.
fn yolo_conv(cin: usize, cout: usize, side: usize, stride: usize) -> ConvGeom {
    ConvGeom { batch: 1, h: side, w: side, cin, cout, kh: 3, kw: 3, stride, pad: 1 }
}

/// The convolutions YOLO26-x tunes, against [`ConvGeom::tiles`] — which lets
/// `M` be ragged, so a 20x20 output tiles where the GEMM's own rule would not.
///
/// What is pinned is the contract: the set is non-empty, every tile in it is
/// distinct, and a kernel that pays per trip is offered strips deeper than the
/// one the GEMM would take. The *order* is a coverage heuristic, not a
/// prediction — on this device measurement still prefers a tile the ranking
/// puts mid-list, and calibrating that away against one part's numbers would
/// only be a tile table written less legibly.
#[test_case(768, 768, 80, 2; "768 channels stride 2, 80x80 in")]
#[test_case(384, 384, 160, 2; "384 channels stride 2, 160x160 in")]
#[test_case(768, 768, 40, 2; "768 channels stride 2, 40x40 in")]
#[test_case(768, 768, 20, 1; "768 channels, 20x20")]
#[test_case(192, 192, 40, 1; "192 channels, 40x40")]
fn the_convolution_is_offered_deep_strips(cin: usize, cout: usize, side: usize, stride: usize) {
    let geom = yolo_conv(cin, cout, side, stride);
    let (m, _, n) = geom.mkn();
    let top = ranked(&rtx_3060(), TripCost::PerStripRow, (m, n), 8, |cfg| geom.tiles(cfg));
    assert!(!top.is_empty(), "{cin}->{cout} must tile");
    let mut distinct = top.clone();
    distinct.dedup();
    assert_eq!(distinct.len(), top.len(), "the tuner must not be handed the same tile twice: {top:?}");
    assert!(
        top.iter().any(|&(_, _, k_step)| k_step >= 64),
        "a kernel that pays per trip must be offered a strip deeper than the GEMM's 32: {top:?}"
    );
}

/// The same device and the same shapes, for a kernel that does not pay per
/// trip: the strip depth stops being the first thing the ranking spends on.
#[test_case(4096, 1024, 6144; "qwen3 gate/up")]
#[test_case(1024, 3072, 1024; "qwen3 down")]
fn the_gemm_is_offered_the_widest_tile_that_covers(m: usize, k: usize, n: usize) {
    let top = ranked(&rtx_3060(), TripCost::Free, (m, n), 4, |cfg| cfg.tiles(m, k, n));
    let (bm, bn, _) = top[0];
    assert!(bm * bn >= 128 * 64, "a covering GEMM grid should take the widest tile, got {top:?}");
}

/// Nothing the model offers may exceed what the device allows.
#[test]
fn every_candidate_fits_the_device() {
    let budget = rtx_3060();
    let bytes = DType::Float16.bytes();
    for &(m, k, n) in &[(1600usize, 9 * 768usize, 768usize), (6400, 9 * 384, 384), (4096, 1024, 6144)] {
        let tiles = budget.ranked(&NT_128X64, bytes, TripCost::PerStripRow, (m, n), 16, |cfg| cfg.tiles(m, k, n));
        assert!(!tiles.is_empty(), "{m}x{k}x{n} has no candidate");
        for cfg in tiles {
            assert!(budget.fits(&cfg, bytes), "{cfg:?} does not fit the device it was generated for");
            assert!(cfg.shared_bytes(bytes) <= budget.limits.shared_per_workgroup, "{cfg:?} overruns shared memory");
            assert!(
                (cfg.warps_m * cfg.warps_n) * budget.wave_size <= budget.limits.max_threads,
                "{cfg:?} overruns the workgroup"
            );
        }
    }
}

/// A shape no tile divides yields nothing rather than something that would fail
/// to launch.
#[test]
fn an_untileable_shape_yields_no_candidate() {
    let budget = rtx_3060();
    let (m, k, n) = (1000usize, 17usize, 33usize);
    let tiles: Vec<GemmCfg> =
        budget.ranked(&NT_128X64, 2, TripCost::PerStripRow, (m, n), 8, |cfg| cfg.tiles(m, k, n)).into_iter().collect();
    assert!(tiles.is_empty(), "expected no candidate for {m}x{k}x{n}, got {tiles:?}");
}
