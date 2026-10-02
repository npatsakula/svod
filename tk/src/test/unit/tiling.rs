//! The tile model against the tiles measurement actually picked.
//!
//! The model is not asked to name the winner — [`crate::tune`] measures that —
//! but the winner has to be among the few candidates it offers, or the search
//! can never reach it. These cases pin that against an RTX 3060 (sm_86) and the
//! convolution shapes YOLO26-x tunes on it, whose per-candidate times were
//! measured on the hardware.

use std::time::Duration;

use svod_dtype::DType;
use test_case::test_case;

use crate::kernels::conv::ConvGeom;
use crate::kernels::gemm::{GemmCfg, NT_128X64};
use crate::kernels::tiling::{TileBudget, Trial, TripCost, agreement, pack};
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

/// An RX 9070 XT (gfx1201): 64 CUs, 64 KiB of LDS a workgroup and a CU, the
/// register file unreported, a 16-wide WMMA fragment on a 32-lane wave.
fn rx_9070_xt() -> TileBudget {
    TileBudget {
        wave_size: 32,
        limits: WorkgroupLimits {
            max_threads: 1024,
            shared_per_workgroup: 64 * 1024,
            shared_per_cu: 64 * 1024,
            registers_per_cu: None,
        },
        mma_edge: 16,
        blocks_wanted: 64,
        resident_floor: 2,
    }
}

/// Every tile the walk can reach for `g`: the seeds and their transitive
/// one-step neighbours, under the kernel's own tiling rule.
fn reachable(budget: &TileBudget, g: &ConvGeom) -> Vec<GemmCfg> {
    let (m, _, n) = g.mkn();
    let mut tiles: Vec<GemmCfg> =
        budget.ranked(&NT_128X64, 2, TripCost::PerStripRow, (m, n), usize::MAX, |cfg| g.tiles(cfg)).to_vec();
    let mut next = 0;
    while next < tiles.len() {
        for cfg in budget.neighbours(&tiles[next], 2, |cfg| g.tiles(cfg)) {
            if !tiles.contains(&cfg) {
                tiles.push(cfg);
            }
        }
        next += 1;
    }
    tiles
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

/// Every step off a tile lands on a tile: the walk may not produce a strip
/// shallower than the matrix core's K edge, a block the wave grid does not
/// divide, or anything the device cannot launch. A `k_step` under the edge is
/// not a smaller tile, it is not a tile — the kernel asserts on it.
#[test]
fn no_step_leaves_the_lattice() {
    let budget = rtx_3060();
    let bytes = DType::Float16.bytes();
    let (m, k, n) = (4096usize, 1024usize, 6144usize);
    let seeds = budget.ranked(&NT_128X64, bytes, TripCost::Free, (m, n), 8, |cfg| cfg.tiles(m, k, n));
    assert!(!seeds.is_empty(), "the GEMM must have somewhere to start");
    let mut frontier: Vec<GemmCfg> = seeds.into_iter().collect();
    for _ in 0..3 {
        let next: Vec<GemmCfg> =
            frontier.iter().flat_map(|cfg| budget.neighbours(cfg, bytes, |c| c.tiles(m, k, n))).collect();
        for cfg in &next {
            assert!(cfg.k_step.is_multiple_of(budget.mma_edge), "{cfg:?} has a strip under the core's K edge");
            assert!(cfg.reg_m().is_multiple_of(budget.mma_edge), "{cfg:?} has a fragment-ragged M");
            assert!(cfg.reg_n().is_multiple_of(budget.mma_edge), "{cfg:?} has a fragment-ragged N");
            assert!(budget.fits(cfg, bytes), "{cfg:?} does not fit the device it was stepped on");
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
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

/// The one tile class the convolution kernel computes wrong — a wave holding a
/// single 16x16 accumulator over a 16-deep strip — is neither seeded nor
/// reached. On gfx1201 all 480 such tiles the walk could reach for YOLO26-m's
/// shapes returned garbage and one of them won the search for the `64→64 k3
/// @80²` bodies; `every_lattice_tile_matches_the_graph_gpu` is the sweep that
/// found it, and this is what keeps it out of the walk on every device.
#[test_case(yolo_conv(64, 64, 80, 1); "m bodies, where the search picked one")]
#[test_case(yolo_conv(64, 64, 160, 2); "n backbone.3, where the store held one")]
#[test_case(yolo_conv(96, 96, 80, 1); "x bodies")]
#[test_case(yolo_conv(512, 64, 20, 1); "m head at 20, a starved grid")]
fn the_single_fragment_tile_is_never_reached(g: ConvGeom) {
    let single = |cfg: &GemmCfg| cfg.reg_m() == 16 && cfg.reg_n() == 16 && cfg.k_step == 16;
    for budget in [rx_9070_xt(), rtx_3060()] {
        let tiles = reachable(&budget, &g);
        assert!(!tiles.is_empty(), "the walk has somewhere to start");
        let reached: Vec<_> = tiles.iter().filter(|cfg| single(cfg)).collect();
        assert!(reached.is_empty(), "{reached:?}");
        // The step onto it from its nearest neighbour is the one refused: every
        // other move from that tile still stands.
        let from = GemmCfg { block_m: 32, block_n: 32, warps_m: 2, warps_n: 2, acc_m: 1, k_step: 32, ..NT_128X64 };
        let moves = budget.neighbours(&from, 2, |cfg| g.tiles(cfg));
        assert!(moves.iter().all(|cfg| !single(cfg)), "{moves:?}");
        assert!(!moves.is_empty(), "the other moves from that tile still stand");
        // The deeper strip stays on offer wherever the channels fill it.
        assert_eq!(moves.iter().any(|cfg| cfg.k_step == 64), g.cin.is_multiple_of(64), "{moves:?}");
    }
}

/// A tile whose time and answer are scripted, so the search's own walk runs
/// without a device.
struct Scripted {
    ns: u64,
    output: Option<Vec<f32>>,
}

impl Trial for Scripted {
    fn time(&self) -> Option<Duration> {
        Some(Duration::from_nanos(self.ns))
    }

    fn output(&self) -> Option<Vec<f32>> {
        self.output.clone()
    }
}

/// The answer a right tile computes, off by up to `rounding` of each value.
fn answer(rounding: f32) -> Vec<f32> {
    (0..256).map(|i| (i as f32 * 0.37).sin() * (1.0 + rounding * ((i % 7) as f32 - 3.0) / 3.0)).collect()
}

/// m's `64→64 k3 @80²` bodies on gfx1201, where a wrong tile once won, and
/// the tiles the walk starts from.
fn m_bodies() -> (TileBudget, ConvGeom, Vec<GemmCfg>) {
    let (budget, g) = (rx_9070_xt(), yolo_conv(64, 64, 80, 1));
    let (m, _, n) = g.mkn();
    let seeds = budget.ranked(&NT_128X64, 2, TripCost::PerStripRow, (m, n), 4, |cfg| g.tiles(cfg)).to_vec();
    (budget, g, seeds)
}

/// What a right tile costs: fixed by its config, never the 1 ns of `walk`'s odd one.
fn right_ns(cfg: &GemmCfg) -> u64 {
    1000 + (pack(cfg) % 997) as u64
}

/// The walk from `seeds` where every tile answers right, each with rounding of
/// its own (up to 4e-3, above the 2.5e-3 the sweep saw), except `odd`: the
/// fastest of all, answering `odd_output`.
fn walk(seeds: &[GemmCfg], odd: GemmCfg, odd_output: Option<Vec<f32>>) -> Option<(GemmCfg, u64)> {
    let (budget, g, _) = m_bodies();
    budget.search(
        seeds,
        2,
        agreement(&DType::Float16),
        |cfg| g.tiles(cfg),
        |cfg| {
            Some(match cfg == odd {
                true => Scripted { ns: 1, output: odd_output.clone() },
                false => Scripted { ns: right_ns(&cfg), output: Some(answer((pack(&cfg) % 5) as f32 * 1e-3)) },
            })
        },
    )
}

fn garbage() -> Vec<f32> {
    answer(0.0).iter().map(|v| 0.2 - 0.7 * v).collect()
}

/// A tile that computes garbage faster than anything right never wins: the
/// seeds agree on the answer without it and it is dropped. Answering right, the
/// same tile wins, so its output is all that sinks it.
#[test]
fn a_fast_wrong_seed_never_wins() {
    let (_, _, seeds) = m_bodies();
    assert!(seeds.len() >= 3, "a majority needs rivals: {seeds:?}");
    for odd in seeds.clone() {
        let won = walk(&seeds, odd, Some(garbage())).expect("the right seeds agree");
        assert!(won.0 != odd && won.1 > 1, "{odd:?} won: {won:?}");
        assert_eq!(walk(&seeds, odd, Some(answer(0.0))), Some((odd, 1)));
    }
}

/// The seeds' answer holds for the rest of the walk: a wrong tile one step from
/// the fastest seed, where the walk goes next, is dropped the same way.
#[test]
fn a_fast_wrong_tile_met_later_is_dropped() {
    let (budget, g, seeds) = m_bodies();
    let fastest = *seeds.iter().min_by_key(|cfg| right_ns(cfg)).expect("seeds");
    let odd = budget
        .neighbours(&fastest, 2, |cfg| g.tiles(cfg))
        .into_iter()
        .find(|cfg| !seeds.contains(cfg))
        .expect("a step off the seeds");
    let won = walk(&seeds, odd, Some(garbage())).expect("the right seeds agree");
    assert!(won.0 != odd && won.1 > 1, "{odd:?} won: {won:?}");
    assert_eq!(walk(&seeds, odd, Some(answer(0.0))), Some((odd, 1)));
}

/// A NaN agrees with nothing, itself included — not even as the seed the
/// ranking put first, which a tie would otherwise favour — and an output that
/// cannot be read back is no answer either.
#[test_case(Some(vec![f32::NAN; 256]); "NaN")]
#[test_case(Some(vec![f32::INFINITY; 256]); "infinity")]
#[test_case(None; "unreadable")]
fn an_output_that_is_no_number_never_wins(odd_output: Option<Vec<f32>>) {
    let (_, _, seeds) = m_bodies();
    let won = walk(&seeds, seeds[0], odd_output).expect("the other seeds agree");
    assert!(won.0 != seeds[0] && won.1 > 1, "{won:?}");
}

/// Seeds that agree on nothing leave no way to tell a right tile from a wrong
/// one: the search gives up and the caller's static tile stands.
#[test]
fn seeds_that_cannot_agree_end_the_search() {
    let (budget, g, seeds) = m_bodies();
    let found = budget.search(
        &seeds,
        2,
        agreement(&DType::Float16),
        |cfg| g.tiles(cfg),
        |cfg| {
            let phase = seeds.iter().position(|seed| *seed == cfg).unwrap_or(seeds.len()) as f32;
            Some(Scripted {
                ns: right_ns(&cfg),
                output: Some((0..256).map(|i| (i as f32 * 0.37 + phase).sin()).collect()),
            })
        },
    );
    assert_eq!(found, None);
}

/// The line between one answer and two sits above what rounding moves two right
/// tiles apart (the sweep: within 2.5e-3 of the graph each, so 5e-3 of each
/// other) and below the wrong tile class (0.78 and up), for every dtype the
/// kernel stores, and it widens with a coarser one.
#[test]
fn the_agreement_line_sits_between_rounding_and_the_wrong_class() {
    for dtype in [DType::Float16, DType::BFloat16, DType::Float32] {
        let line = agreement(&dtype);
        assert!(line > 5e-3 && line < 0.78, "{dtype:?}: {line}");
    }
    assert!(agreement(&DType::BFloat16) > agreement(&DType::Float16));
}
