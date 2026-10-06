//! The tile model against the tiles measurement actually picked.
//!
//! The model is not asked to name the winner — [`crate::tune`] measures that —
//! but the winner has to be among the few candidates it offers, or the search
//! can never reach it. These cases pin that against an RTX 3060 (sm_86) and the
//! convolution shapes YOLO26-x tunes on it, whose per-candidate times were
//! measured on the hardware.

use std::sync::Arc;
use std::time::Duration;

use svod_dtype::{AmdArch, CudaArch, DType, GpuArch};
use svod_ir::UOp;
use test_case::test_case;

use crate::kernels::conv::{ConvGeom, build_conv, stored_tile};
use crate::kernels::gemm::{Epilogue, GemmCfg, NT_128X64};
use crate::kernels::tiling::{TileBudget, Trial, TripCost, pack};
use crate::target::WorkgroupLimits;

const SM86: GpuArch = GpuArch::Cuda(CudaArch::from_compute_capability(8, 6));
const RDNA4: GpuArch = GpuArch::Amd(AmdArch::Gfx1201);

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

/// Every tile the walk can reach keeps its accumulator read inside the K loop,
/// on both devices, checked off the GPU. Where the read lands depends only on
/// which of `mma`'s height, width and K loops are trip-1 (a trip-1 loop folds
/// away), so one tile of each of those classes is built. Among them is the
/// single-fragment tile (one 16x16 accumulator over a 16-deep strip), whose read
/// once hoisted out of the loop and computed garbage fast enough to win the
/// search for m's `64→64 k3 @80²` bodies.
#[test]
fn every_lattice_tile_keeps_its_accumulator_in_the_k_loop() {
    let shapes =
        [yolo_conv(64, 64, 80, 1), yolo_conv(64, 64, 160, 2), yolo_conv(96, 96, 80, 1), yolo_conv(512, 64, 20, 1)];
    for (budget, arch) in [(rx_9070_xt(), RDNA4), (rtx_3060(), SM86)] {
        let mut classes: Vec<([bool; 3], ConvGeom, GemmCfg)> = Vec::new();
        for g in shapes {
            for cfg in reachable(&budget, &g) {
                let class = [cfg.reg_m(), cfg.reg_n(), cfg.k_step].map(|edge| edge == budget.mma_edge);
                if classes.iter().all(|(seen, ..)| *seen != class) {
                    classes.push((class, g, cfg));
                }
            }
        }
        assert!(classes.iter().any(|(class, ..)| *class == [true; 3]), "{arch:?} reaches the single-fragment tile");
        for (class, g, cfg) in classes {
            let escaped = super::escaped_accumulator_reads(&super::lowered_program(conv_sink(arch, g, cfg), arch));
            assert!(escaped.is_empty(), "{arch:?} {class:?} {cfg:?}: {escaped:#?}");
        }
    }
}

/// A stored search result decodes to its tile only while the walk could still
/// reach it: a lattice tile the device admits and the kernel tiles.
#[test]
fn a_stored_tile_outside_the_lattice_is_a_miss() {
    let (budget, g) = (rx_9070_xt(), yolo_conv(96, 96, 80, 1));
    let base = GemmCfg { l2_swizzle: false, ..NT_128X64 };
    let stored = |cfg: GemmCfg| stored_tile(&budget, &g, &base, 2, pack(&cfg));
    let deepened = |cfg: &GemmCfg| GemmCfg { k_step: 64, ..*cfg };
    let reached = reachable(&budget, &g)
        .into_iter()
        .find(|cfg| cfg.k_step == 32 && budget.admits(&deepened(cfg), 2))
        .expect("a 32-deep tile whose 64-deep strip still fits");
    assert_eq!(stored(reached).map(|cfg| pack(&cfg)), Some(pack(&reached)), "a tile the walk reaches");
    assert_eq!(stored(GemmCfg { k_step: 8, ..reached }), None, "a strip under the core's K edge");
    let huge = GemmCfg { block_m: 256, block_n: 256, warps_m: 1, warps_n: 1, acc_m: 1, ..reached };
    assert!(!budget.fits(&huge, 2) && stored(huge).is_none(), "a tile past the device's limits");
    assert_eq!(stored(deepened(&reached)), None, "a strip 96 channels do not divide into");
    assert_eq!(stored_tile(&budget, &g, &base, 2, pack(&reached) | 1 << 30), None, "bits past the lattice's");
}

/// `build_conv`'s graph of `cfg` on `arch` over placeholder buffers, with the
/// unsearched fields the walk's own seeds take.
fn conv_sink(arch: GpuArch, g: ConvGeom, cfg: GemmCfg) -> Arc<UOp> {
    let cfg = GemmCfg { l2_swizzle: false, ..cfg };
    let caps = crate::ArchCaps::for_arch(arch);
    let (m, k, n) = g.mkn();
    let dt = DType::Float16;
    let bufs = [m * n, g.batch * g.h * g.w * g.cin, n * k, n]
        .into_iter()
        .map(|size| UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, size, dt.clone()))
        .collect();
    let ker = crate::Kernel::new("conv2d_nhwc", g.grid_dims(&cfg), cfg.threads(caps.wave_size), bufs, caps);
    build_conv(&ker, g, cfg, dt, Epilogue::BiasAct { bias: (), residual: None, act: true });
    ker.finish(cfg.acc_m)
}

/// A tile whose time is scripted, so the search's own walk runs without a device.
struct Scripted(Option<u64>);

impl Trial for Scripted {
    fn time(&self) -> Option<Duration> {
        self.0.map(Duration::from_nanos)
    }
}

/// m's `64→64 k3 @80²` bodies on gfx1201, where a wrong tile once won, and
/// the tiles the walk starts from.
fn m_bodies() -> (TileBudget, ConvGeom, Vec<GemmCfg>) {
    let (budget, g) = (rx_9070_xt(), yolo_conv(64, 64, 80, 1));
    let (m, _, n) = g.mkn();
    let seeds = budget.ranked(&NT_128X64, 2, TripCost::PerStripRow, (m, n), 4, |cfg| g.tiles(cfg)).to_vec();
    (budget, g, seeds)
}

/// What a tile costs in `walk`: fixed by its config, never the 1 ns of a scripted winner.
fn right_ns(cfg: &GemmCfg) -> u64 {
    1000 + (pack(cfg) % 997) as u64
}

/// The walk over m's bodies, where `time` scripts each tile's device time.
fn walk(time: impl Fn(&GemmCfg) -> Option<u64>) -> Option<(GemmCfg, u64)> {
    let (budget, g, seeds) = m_bodies();
    budget.search(&seeds, 2, |cfg| g.tiles(cfg), |cfg| Some(Scripted(time(&cfg))))
}

/// The walk keeps the fastest tile it times, a step off the seeds included, and
/// never one it could not time, even the seed that would otherwise be fastest.
#[test]
fn the_walk_keeps_the_fastest_tile_it_times() {
    let (budget, g, seeds) = m_bodies();
    let fastest = *seeds.iter().min_by_key(|cfg| right_ns(cfg)).expect("seeds");
    let step = budget
        .neighbours(&fastest, 2, |cfg| g.tiles(cfg))
        .into_iter()
        .find(|cfg| !seeds.contains(cfg))
        .expect("a step off the seeds");
    assert_eq!(walk(|cfg| Some(if *cfg == step { 1 } else { right_ns(cfg) })), Some((step, 1)));
    let (won, _) = walk(|cfg| (*cfg != fastest).then(|| right_ns(cfg))).expect("the other seeds time");
    assert_ne!(won, fastest, "an untimed tile never wins");
}
