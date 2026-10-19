//! The op layer's decisions on the host: which plan each shape, dtype and
//! target gets, and which requests are errors rather than fallbacks.

use std::sync::Arc;

use svod_dtype::{AmdArch, DType, DeviceSpec, GpuArch, MetalFamily};
use svod_ir::SInt;
use svod_tensor::{Tensor, Variable};
use test_case::test_case;

use super::conv::geom;
use super::{costs, first_eight, planner};
use crate::atoms::{Costs, Device, Target, sm86};
use crate::build::BF16 as BF16T;
use crate::kernels::Batch;
use crate::kernels::attention::{FaCfg, Forms};
use crate::kernels::conv::ConvCfg;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, Scale, gemm};
use crate::kernels::rows::NormCfg;
use crate::ops::attention::{descend, split_ranked};
use crate::ops::config::{self, Planner, Run, SHORTLIST, Tiled, gemm_cfg};
use crate::ops::shape::{self, Extent, Fallback, Plan, extent};
use crate::ops::{self as tk, Act, Attn, Cache, Error, KeyMask, Linear, Qkv};

const BF16: DType = DType::BFloat16;

fn ext(dims: &[usize]) -> Extent {
    Extent { dims: dims.to_vec(), var: false }
}

/// The candidates of a plan that takes the kernel.
fn kernel_plan<C>(plan: Plan<C>) -> Arc<[C]> {
    match plan {
        Plan::Kernel(list) => list,
        Plan::Graph(why) => panic!("graph: {why:?}"),
    }
}

/// The untuned pick of a plan: its first candidate.
fn first<C: Clone>(plan: Plan<C>) -> Result<C, Fallback> {
    match plan {
        Plan::Kernel(c) => Ok(c[0].clone()),
        Plan::Graph(why) => Err(why),
    }
}

fn rdna3() -> Target {
    Target { sms: Some(48), ..Target::for_arch(GpuArch::Amd(AmdArch::Gfx1100)) }
}

/// A target with a matrix core and no tables.
fn apple() -> Target {
    Target { sms: Some(10), ..Target::for_arch(GpuArch::Metal(MetalFamily::Apple(9))) }
}

// ---- linear ------------------------------------------------------------------------

/// The config measured fastest at 4096³ on the 3060 (25.4 TFLOP/s): the
/// 128×128 tile and three 32-deep slots over eight warps.
const BIG_MEASURED: GemmCfg = GemmCfg { tile: [128, 128, 32], stages: 3, warps: [2, 4], group_m: 8, unroll: false };

/// A list the planner offers for `k` on `target`: not empty, each config
/// once, lowering there with its ring in shared memory and dividing `k`;
/// its untested shortlist one pipeline per output tile and warp grid.
fn assert_valid(list: &[GemmCfg], target: &Target, k: usize, gated: bool) {
    assert!(!list.is_empty(), "{list:?}");
    for (i, c) in list.iter().enumerate() {
        assert!(c.fits(target) && k.is_multiple_of(c.tile[2]), "{c:?}");
        assert!(c.smem_bytes(gated) <= target.smem_bytes, "{c:?}");
        assert!(!list[..i].contains(c), "{c:?} twice");
    }
    let short = first_eight(list);
    for (i, c) in short.iter().enumerate() {
        assert!(!short[..i].iter().any(|o| o.tile[..2] == c.tile[..2] && o.warps == c.warps), "{short:?}");
    }
}

/// Every shape the kernel takes on sm_86 plans a valid list (a reduction dim
/// off 32 takes 16-deep steps); the rest keep the graph for their reason.
/// A config whose accumulators and one matrix-core depth of operands pass
/// a lane's registers never compiles clean, so the ranking drops it: a
/// 256×96 tile over two warps (384 accumulators a lane on sm_86), never a
/// 64×64 one over four, gated or not.
#[test_case(sm86(), false; "sm_86")]
#[test_case(sm86(), true; "sm_86 gated")]
#[test_case(crate::test::unit::targets::amd(AmdArch::Gfx1201, 64), false; "gfx1201")]
fn the_ranking_drops_what_a_lane_cannot_hold(target: Target, gated: bool) {
    let limit = target.occupancy.registers;
    let ranked = planner(target.clone()).gemm_ranked(1, 704, 1536, 512, gated);
    let halves = if gated { 2 } else { 1 };
    assert!(ranked.iter().all(|c| c.live_registers(&target, halves).unwrap() <= limit));
    let has = |tile: [usize; 3], warps: [u32; 2]| ranked.iter().any(|c| c.tile == tile && c.warps == warps);
    assert_eq!((has([256, 96, 32], [2, 1]), has([64, 64, 32], [2, 2])), (false, true));
}

#[test_case(&[4096, 4096], 4096, false, Ok(()); "large grid")]
#[test_case(&[4096, 4096], 4096, true, Ok(()); "gated")]
#[test_case(&[8, 256, 512], 2048, false, Ok(()); "a static batch counts toward the grid")]
#[test_case(&[2048, 512], 1024, false, Ok(()); "half the big grid")]
#[test_case(&[8, 37, 512], 512, false, Ok(()); "medium grid")]
#[test_case(&[37, 64], 96, false, Ok(()); "few rows")]
#[test_case(&[37, 48], 96, false, Ok(()); "k a multiple of 16 only")]
#[test_case(&[37, 40], 96, false, Err(Fallback::Config); "k off every bk")]
#[test_case(&[37, 64], 100, false, Err(Fallback::Shape); "n not a multiple of 8")]
#[test_case(&[0, 64], 96, false, Err(Fallback::Shape); "no rows")]
fn linear_plans(x: &[usize], n: usize, gated: bool, want: Result<(), Fallback>) {
    let plan = shape::linear(Some(&planner(sm86())), &[BF16, BF16], Some(&ext(x)), n, gated);
    match plan {
        Plan::Kernel(list) => assert_valid(&list, &sm86(), *x.last().unwrap(), gated),
        Plan::Graph(why) => assert_eq!(Err(why), want),
    }
}

fn gemm_list(x: &Extent, n: usize, gated: bool) -> Arc<[GemmCfg]> {
    kernel_plan(shape::linear(Some(&planner(sm86())), &[BF16, BF16], Some(x), n, gated))
}

/// The 4096³ ranking on sm_86 is valid.
#[test]
fn large_gemm_candidates() {
    assert_valid(&gemm_list(&ext(&[4096, 4096]), 4096, false), &sm86(), 4096, false);
}

/// Nemotron's projections (704 rows under a batch variable of capacity 1)
/// rank more than eight configs, so the shortlist spans eight tiles and
/// grids.
#[test_case(512, 512, false; "qkv-sized out")]
#[test_case(1536, 512, false; "fused qkv")]
#[test_case(2048, 512, false; "ffn up")]
#[test_case(512, 2048, false; "ffn down")]
#[test_case(2048, 512, true; "gated up")]
fn small_m_gemm_candidates(n: usize, k: usize, gated: bool) {
    let x = Extent { dims: vec![1, 704, k], var: true };
    let list = gemm_list(&x, n, gated);
    assert!(list.len() >= 8, "{list:?}");
    assert_valid(&list, &sm86(), k, gated);
}

fn gfx1201() -> Target {
    Target { sms: Some(64), ..Target::for_arch(GpuArch::Amd(AmdArch::Gfx1201)) }
}

/// A gated weight stages a B tile twice the config's width and holds two
/// accumulators: by the layout count the 128×128 tile over eight warps
/// holds 216 registers a lane gated where its half-width form holds 128, and
/// its 64-deep form more than gfx1201's 256, whose two-slot ring of a
/// 384-wide fill is past the 64 KB of LDS besides. The ranking keeps steps
/// at least two matrix-core depths deep.
#[test]
fn a_gated_gemm_is_a_double_width_problem() {
    let target = gfx1201();
    let eight = gemm_cfg([128, 128, 32], 2, [2, 4], true);
    let half = gemm_cfg([128, 64, 32], 2, [2, 4], true);
    let deep = gemm_cfg([128, 128, 64], 2, [2, 4], true);
    assert_eq!((eight.registers(&target, 2), half.registers(&target, 2)), (Some(216), Some(128)));
    assert!(deep.registers(&target, 2).is_some_and(|r| r > 256), "{:?}", deep.registers(&target, 2));
    let gated = planner(target.clone()).gemm_ranked(1, 4096, 3072, 1024, true);
    assert_valid(&gated, &target, 1024, true);
    assert!(gated.iter().all(|c| c.tile[2] >= 32), "{gated:?}");
    let p = config::Problem { batches: 1, m: 4096, n: 3072, k: 1024, halves: 2 };
    assert!(deep.smem_bytes(true) > target.smem_bytes && p.admit(&planner(target), &deep).is_none());
}

/// The layout count over-counts gfx11's replicated WMMA fragments: at
/// `d = 128` it puts 64×16 and 128×16 past the 256 registers a lane holds,
/// yet both compile without spilling, rolled. So no count admits: both are
/// ranked, and the compiled kernel decides.
#[test]
fn no_register_count_admits() {
    let target = rdna3();
    let ranked = planner(target.clone()).attention_ranked(8, 1500, 1500, 128, false);
    for (bq, bkv) in [(64, 16), (128, 16)] {
        let over = FaCfg::new(bq, bkv, 2).registers(&target, 128).is_some_and(|r| r > 256);
        assert!(over, "{bq}×{bkv} counts past the cap");
        assert!(ranked.iter().any(|c| (c.bq, c.bkv) == (bq, bkv)), "{bq}×{bkv} not in {ranked:?}");
    }
}

/// The SM time of `rounds` rounds of `actual` resident blocks of `waves`
/// waves, each `trips` trips of `work`, on sm_86 under `costs`: blocks
/// filling the SMs `rounds` times over, each holding `1/actual` of an SM's
/// shared memory (at most twelve resident: sm_86's 48 waves over four).
fn pipelined(costs: Costs, [rounds, trips, actual, waves, stages]: [usize; 5], work: f64) -> f64 {
    let planner = Planner::new(Device { target: sm86(), costs });
    let (blocks, smem) = (rounds * actual * planner.sms(), sm86().smem_bytes / actual);
    planner.pipelined(Run { blocks, smem, regs: 0, waves, trips, work, trip: 0.0, stages })
}

/// The busiest SM issues `⌈blocks / sms⌉` blocks' work, not whole rounds of
/// what it holds: nine blocks an SM over rounds of four cost nine blocks.
#[test]
fn an_sm_issues_its_share_of_blocks() {
    let flat = Costs { latency: 0.0, ..costs() };
    let planner = Planner::new(Device { target: sm86(), costs: flat });
    let sms = planner.sms();
    let run = |blocks| Run {
        blocks,
        smem: sm86().smem_bytes / 4,
        regs: 0,
        waves: 4,
        trips: 1,
        work: 100.0,
        trip: 0.0,
        stages: 2,
    };
    assert_eq!(planner.pipelined(run(9 * sms)), 900.0);
    assert_eq!(planner.pipelined(run(9 * sms + 1)), 1000.0);
}

/// A trip's fixed overhead is shared by the blocks an SM holds at once,
/// its work is not.
#[test_case(1; "one block an SM")]
#[test_case(4; "four")]
fn a_trip_overhead_is_shared_by_resident_blocks(resident: usize) {
    let flat = Costs { latency: 0.0, ..costs() };
    let planner = Planner::new(Device { target: sm86(), costs: flat });
    let blocks = resident * planner.sms();
    let smem = sm86().smem_bytes / resident;
    let run = |work, trip| Run { blocks, smem, regs: 0, waves: 4, trips: 1, work, trip, stages: 2 };
    let base = planner.pipelined(run(100.0, 0.0));
    assert_eq!(planner.pipelined(run(100.0, 40.0)) - base, 40.0);
}

/// A fill's byte is DRAM's where no cache was measured or the distinct
/// operands overflow it, the cache's for the re-reads where they fit, and in
/// between as much nearer DRAM as they overflow three quarters of it.
#[test]
fn a_fill_is_priced_by_where_its_bytes_come_from() {
    let mb = (1 << 20) as f64;
    let with = |l2_bytes| {
        let c = Costs { global_byte: 0.08, l2_byte: 0.01, l2_bytes, ..costs() };
        Planner::new(Device { target: sm86(), costs: c })
    };
    let (none, cached) = (with(0.0), with(4.0 * mb));
    assert_eq!(none.fill_byte(mb, 100.0 * mb), 0.08);
    // 1 MB read 100 times over: one byte in a hundred from DRAM.
    assert!((cached.fill_byte(mb, 100.0 * mb) - (0.01 * 0.99 + 0.08 * 0.01)).abs() < 1e-12);
    // Past 1.5 × 3 MB every byte comes from DRAM.
    assert_eq!(cached.fill_byte(5.0 * mb, 100.0 * mb), 0.08);
    let edge: Vec<f64> = [2.0, 3.0, 3.5, 4.0, 4.5].iter().map(|&u| cached.fill_byte(u * mb, 100.0 * mb)).collect();
    assert!(edge.windows(2).all(|w| w[0] <= w[1]), "{edge:?}");
    // A fill that reads each byte once pays DRAM whatever the cache.
    assert_eq!(cached.fill_byte(mb, mb), 0.08);
}

/// The ranking's front spreads over the band's block tiles: a block tile's
/// second tile comes after every class's first, and every config is kept.
#[test]
fn the_ranking_leads_with_a_tile_of_each_block() {
    let cfg = |tile: [usize; 3], warps: [u32; 2], stages| GemmCfg { tile, stages, warps, group_m: 8, unroll: false };
    let ranked = vec![
        (100.0, cfg([128, 128, 64], [2, 2], 3)),
        (100.0, cfg([128, 128, 64], [2, 2], 2)),
        (101.0, cfg([128, 128, 64], [1, 4], 3)),
        (102.0, cfg([128, 128, 32], [2, 2], 3)),
        (110.0, cfg([64, 64, 32], [2, 2], 3)),
        (120.0, cfg([64, 128, 32], [1, 2], 2)),
        (200.0, cfg([32, 64, 32], [1, 1], 2)),
    ];
    let spread = config::spread(ranked.clone(), |c| [c.tile[0], c.tile[1]]);
    let blocks: Vec<[usize; 2]> = config::distinct(&spread).take(3).map(|c| [c.tile[0], c.tile[1]]).collect();
    assert_eq!(blocks, [[128, 128], [64, 64], [64, 128]]);
    // A tile's pipelines follow it; the band's second tiles come next; then
    // what lies past the band.
    assert_eq!(spread[..2], [ranked[0].1, ranked[1].1]);
    assert_eq!(spread[spread.len() - 1], ranked[6].1);
    let mut sorted = spread.clone();
    sorted.sort_by_key(|c| format!("{c:?}"));
    let mut want: Vec<GemmCfg> = ranked.iter().map(|(_, c)| *c).collect();
    want.sort_by_key(|c| format!("{c:?}"));
    assert_eq!(sorted, want);
}

/// Without latency a round is its issue work: linear in the work, the
/// resident blocks, the trips and the rounds, whatever the ring.
#[test]
fn without_latency_a_round_is_its_work() {
    let flat = Costs { latency: 0.0, ..costs() };
    let cost = |rounds, trips, actual, work, stages| pipelined(flat, [rounds, trips, actual, 4, stages], work);
    assert_eq!(cost(1, 1, 1, 100.0, 2), 100.0);
    assert_eq!(cost(1, 1, 1, 300.0, 2), 3.0 * cost(1, 1, 1, 100.0, 2));
    assert_eq!(cost(2, 3, 4, 100.0, 3), 24.0 * cost(1, 1, 1, 100.0, 3));
    assert_eq!(cost(1, 1, 1, 100.0, 2), cost(1, 1, 1, 100.0, 3));
}

proptest::proptest! {
    /// A block of fewer waves than the SM's SIMDs leaves SIMDs idle: with
    /// no latency, a one-warp block never beats a block of as many warps as
    /// SIMDs doing the same work, and is that many times slower alone.
    #[test]
    fn a_one_warp_block_never_beats_a_full_one(
        rounds in 1usize..8,
        trips in 1usize..64,
        actual in 1usize..12,
        work in 1.0f64..1e4,
    ) {
        let flat = Costs { latency: 0.0, ..costs() };
        let simds = sm86().occupancy.simds as usize;
        let one = pipelined(flat, [rounds, trips, actual, 1, 2], work);
        let full = pipelined(flat, [rounds, trips, actual, simds, 2], work);
        proptest::prop_assert!(one >= full, "{one} < {full}");
        let alone = pipelined(flat, [rounds, trips, 1, 1, 2], work);
        proptest::prop_assert_eq!(alone, simds as f64 * pipelined(flat, [rounds, trips, 1, simds, 2], work));
    }
}

/// Under a latency no work covers, a round costs the latency its ring
/// leaves, so a register file holding twice the resident blocks halves the
/// rounds and the cost; a deeper ring hides more of it.
#[test]
fn under_latency_resident_blocks_halve_the_cost() {
    let slow = Costs { latency: 1e9, ..costs() };
    let p = config::Problem { batches: 112, m: 128, n: 128, k: 4096, halves: 1 };
    let cfg = gemm_cfg([128, 128, 32], 3, [2, 4], false);
    let full = sm86();
    let starved = Target { occupancy: crate::atoms::Occupancy { waves: 2, ..full.occupancy }, ..sm86() };
    let cost = |target: Target| p.admit(&Planner::new(Device { target, costs: slow }), &cfg).expect("admitted").0;
    assert_eq!(2.0 * cost(full), cost(starved));
    assert!(pipelined(slow, [1, 1, 1, 4, 3], 1.0) < pipelined(slow, [1, 1, 1, 4, 2], 1.0));
}

/// A ring of one stage or none prefetches nothing: the round waits the
/// whole latency, never a division by zero.
#[test_case(0; "no ring")]
#[test_case(1; "one stage")]
fn a_ring_without_prefetch_waits_the_latency(stages: usize) {
    let c = costs();
    assert_eq!(pipelined(c, [1, 1, 1, 4, stages], 0.0), c.latency);
    assert_eq!(pipelined(c, [1, 1, 1, 4, stages], 0.0), pipelined(c, [1, 1, 1, 4, 2], 0.0));
}

/// A ranked config of the shortlist tests: its position, and the tile it
/// is a pipeline of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Pipe {
    at: usize,
    tile: usize,
}

impl config::Tiled for Pipe {
    fn tile(&self) -> Self {
        Pipe { at: 0, ..*self }
    }

    fn twins(&self, ranked: &[Self]) -> Vec<Self> {
        ranked.iter().filter(|p| p.tile == self.tile).copied().collect()
    }
}

/// The tuner's shortlist is the first [`SHORTLIST`] ranked configs whose
/// compiled kernels do not spill, one per tile (its first clean pipeline),
/// in ranked order, compiled in batches of as many until that many tiles
/// pass: a batch past that is never compiled. Where every config that
/// compiles spills, the first [`SHORTLIST`] of those are timed anyway;
/// where nothing compiles there is nothing to judge. `twins` are `(config,
/// the config whose tile it shares)`; `ranked` is in shortlists.
#[test_case(&[], &[], &[], 1.5, Some(&(0..SHORTLIST).collect::<Vec<_>>()), 1; "nothing spills: the first batch")]
#[test_case(&[1, 4, 9], &[], &[], 1.5, Some(&(0..SHORTLIST + 3).filter(|i| ![1, 4, 9].contains(i)).collect::<Vec<_>>()), 2; "spills skipped, order kept")]
#[test_case(&[0, 1, 2], &[], &[], 0.0, Some(&[3, 4]), 1; "a short ranking keeps what passes")]
#[test_case(&[0, 2, 3], &[1], &[], 0.0, Some(&[4]), 1; "a failure is never timed")]
#[test_case(&[0, 1], &[], &[], 0.0, Some(&[0, 1]), 1; "every config spills: they are timed anyway")]
#[test_case(&[0], &[], &[], 0.0, Some(&[0]), 1; "a single spilling config is timed alike")]
#[test_case(&[], &[], &[], 0.0, Some(&[0]), 1; "a single clean config")]
#[test_case(&(0..2 * SHORTLIST).collect::<Vec<_>>(), &[], &[], 2.0, Some(&(0..SHORTLIST).collect::<Vec<_>>()), 2; "every config spills")]
#[test_case(&[0, 1], &[2], &[], 0.0, Some(&[0, 1]), 1; "spilled and failed: the spilled")]
#[test_case(&[], &[0, 1, 2], &[], 0.0, None, 1; "nothing compiles")]
#[test_case(&[], &[], &[], 0.0, None, 0; "nothing ranked")]
#[test_case(&[0], &[], &[(1, 0)], 0.0, Some(&[1, 2]), 1; "a spilling unrolled ring yields to its clean rolled twin")]
#[test_case(&[], &[], &[(1, 0)], 0.0, Some(&[0, 2]), 1; "two clean pipelines of one tile: the first")]
#[test_case(&[], &[], &[(1, 0), (3, 2)], 1.5, Some(&[0, 2].into_iter().chain(4..SHORTLIST + 2).collect::<Vec<_>>()), 2; "tiles, not configs")]
#[test_case(&[0, 1], &[], &[(1, 0)], 0.0, Some(&[0, 1]), 1; "both pipelines spill: both are timed")]
fn the_shortlist_is_the_first_tiles_that_do_not_spill(
    spilled: &[usize],
    failed: &[usize],
    twins: &[(usize, usize)],
    ranked: f64,
    want: Option<&[usize]>,
    batches: usize,
) {
    use crate::tune::Compiled;
    // Short rankings are as long as what they name.
    let ranked = if ranked == 0.0 {
        spilled
            .iter()
            .chain(failed)
            .chain(twins.iter().map(|(c, _)| c))
            .max()
            .map_or(0, |m| m + 1)
            .max(want.map_or(0, |w| w.iter().max().map_or(0, |m| m + 1)))
    } else {
        (ranked * SHORTLIST as f64) as usize
    };
    let tile = |at: usize| twins.iter().find(|(c, _)| *c == at).map_or(at, |(_, t)| *t);
    let ranked: Vec<Pipe> = (0..ranked).map(|at| Pipe { at, tile: tile(at) }).collect();
    let mut asked = vec![];
    let got = config::shortlist(&ranked, |batch| {
        asked.push(batch.to_vec());
        let verdict = |p: &Pipe| match (spilled.contains(&p.at), failed.contains(&p.at)) {
            (_, true) => Compiled::Failed,
            (true, _) => Compiled::Spilled,
            _ => Compiled::Clean,
        };
        batch.iter().map(verdict).collect()
    });
    assert_eq!(got.as_deref(), want);
    assert_eq!(asked.len(), batches, "{asked:?}");
    assert!(asked.iter().flatten().eq(ranked.iter().take(SHORTLIST * batches)), "ranked batches: {asked:?}");
}

/// The first clean config is searched for in ranked batches of eight, the
/// search stopping at the batch that holds one: a spill or a failure ahead
/// of it is skipped, every config spilling yields the first that compiles,
/// and nothing compiling finds nothing. `count` configs are ranked.
#[test_case(&[], &[], 20, Some(0), 1; "the first is clean")]
#[test_case(&[0, 1], &[2], 20, Some(3), 1; "spills and a failure skipped")]
#[test_case(&(0..9).collect::<Vec<_>>(), &[], 20, Some(9), 2; "the second batch")]
#[test_case(&[0, 1, 2], &[], 3, Some(0), 1; "every config spills: the first")]
#[test_case(&[1], &[0, 2], 3, Some(1), 1; "the spilled over the failed")]
#[test_case(&[], &[0, 1, 2], 3, None, 1; "nothing compiles")]
#[test_case(&[], &[], 0, None, 0; "nothing ranked")]
fn the_first_clean_config_is_searched_in_small_batches(
    spilled: &[usize],
    failed: &[usize],
    count: usize,
    want: Option<usize>,
    batches: usize,
) {
    use crate::tune::Compiled;
    let ranked: Vec<Pipe> = (0..count).map(|at| Pipe { at, tile: at }).collect();
    let mut asked = vec![];
    let got = config::first_clean(&ranked, |batch| {
        asked.push(batch.to_vec());
        let verdict = |p: &Pipe| match (spilled.contains(&p.at), failed.contains(&p.at)) {
            (_, true) => Compiled::Failed,
            (true, _) => Compiled::Spilled,
            _ => Compiled::Clean,
        };
        batch.iter().map(verdict).collect()
    });
    assert_eq!(got, want);
    assert_eq!(asked.len(), batches, "{asked:?}");
    assert!(asked.iter().all(|b| b.len() <= 8), "small batches: {asked:?}");
    assert!(asked.iter().flatten().eq(ranked.iter().take(8 * batches)), "ranked batches: {asked:?}");
}

/// A tile's twins are judged whole: every clean one, in order, none set
/// aside for sharing the tile; every one spilling, all of them; nothing
/// compiling, nothing.
#[test_case(&[], &[], 4, Some(&[0, 1, 2, 3]); "all clean")]
#[test_case(&[1], &[3], 4, Some(&[0, 2]); "a spill and a failure dropped")]
#[test_case(&[0, 1, 2], &[], 3, Some(&[0, 1, 2]); "every one spills")]
#[test_case(&[0], &[1], 2, Some(&[0]); "the spilled when nothing is clean")]
#[test_case(&[], &[0, 1], 2, None; "nothing compiles")]
fn twins_are_judged_whole(spilled: &[usize], failed: &[usize], count: usize, want: Option<&[usize]>) {
    use crate::tune::Compiled;
    let twins: Vec<Pipe> = (0..count).map(|at| Pipe { at, tile: 0 }).collect();
    let got = config::clean(&twins, |batch| {
        assert_eq!(batch.len(), count, "one batch");
        let verdict = |p: &Pipe| match (spilled.contains(&p.at), failed.contains(&p.at)) {
            (_, true) => Compiled::Failed,
            (true, _) => Compiled::Spilled,
            _ => Compiled::Clean,
        };
        batch.iter().map(verdict).collect()
    });
    assert_eq!(got.as_deref(), want);
}

/// A GEMM config's twins are the ranking's configs of its block tile and
/// warp grid: every reduction depth, ring depth and unrolling, itself
/// among them, in ranked order.
#[test]
fn a_gemm_tiles_twins_are_its_pipelines() {
    let planner = planner(sm86());
    let ranked = planner.gemm_ranked(1, 704, 1536, 512, false);
    let winner = ranked.iter().find(|c| c.tile[..2] == [64, 64] && c.warps == [2, 2]).copied().unwrap();
    let twins = winner.twins(&ranked);
    assert!(twins.contains(&winner));
    assert!(twins.len() >= 4, "{twins:?}");
    assert!(twins.iter().all(|c| c.tile[..2] == [64, 64] && c.warps == [2, 2]), "{twins:?}");
    assert!(twins.iter().any(|c| c.tile[2] != winner.tile[2]) && twins.iter().any(|c| c.stages != winner.stages));
    let order: Vec<usize> = twins.iter().map(|t| ranked.iter().position(|c| c == t).unwrap()).collect();
    assert!(order.windows(2).all(|w| w[0] < w[1]), "{order:?}");
    assert_eq!(config::distinct_configs(&twins), twins);
}

/// A split convolution's twins keep its split; a split attention's keep its
/// splits and its tuned softmax forms, the ranking's being plain; a norm
/// config is its own only twin.
#[test]
fn variants_keep_their_variant_across_twins() {
    let planner = planner(sm86());
    let ranked: Vec<ConvCfg> = planner.conv_ranked(1, 6400, &geom([80, 80], 64, 64, 3, 1, 1, 1)).to_vec();
    let winner = ConvCfg { split: 3, ..ranked[0] };
    let twins = winner.twins(&ranked);
    assert!(twins.len() >= 2 && twins.iter().all(|c| c.split == 3 && c.gemm.tile() == winner.gemm.tile()), "{twins:?}");
    let ranked = planner.attention_ranked(8, 2048, 2048, 64, false);
    let forms = Forms { raw_max: true, ..Forms::default() };
    let winner = FaCfg { splits: 4, forms, ..ranked[0] };
    let twins = winner.twins(&ranked);
    assert!(twins.len() >= 2, "{twins:?}");
    assert!(
        twins
            .iter()
            .all(|c| (c.bq, c.bkv, c.key_warps, c.splits, c.forms)
                == (winner.bq, winner.bkv, winner.key_warps, 4, forms)),
        "{twins:?}"
    );
    assert!(twins.iter().any(|c| (c.stages, c.unroll) != (winner.stages, winner.unroll)));
    let norm = NormCfg { br: 8 };
    assert_eq!(norm.twins(&[NormCfg { br: 4 }, norm, NormCfg { br: 16 }]), vec![norm]);
}

/// At zero costs every config ties: the ranking is degenerate (fewest
/// registers first) but finite, and still lists what lowers.
#[test]
fn a_ranking_at_zero_costs_is_finite() {
    let planner = Planner::new(Device { target: sm86(), costs: Costs::default() });
    let p = config::Problem { batches: 1, m: 4096, n: 4096, k: 4096, halves: 1 };
    let ranked = p.ranked(&planner);
    assert!(!ranked.is_empty());
    assert!(ranked.iter().all(|c| p.admit(&planner, c).is_some_and(|(cost, _)| cost == 0.0)));
    let a = config::Attention { tiles: 8, t: 100, tk: 100, d: 64, causal: true };
    let ranked = a.ranked(&planner);
    assert!(!ranked.is_empty());
    assert!(ranked.iter().all(|c| a.admit(&planner, c).is_some_and(|(cost, _)| cost == 0.0)));
}

/// Every list is non-empty, duplicate-free, within shared memory and divides `k`.
#[test_case(&[4096, 4096], 4096, false; "large")]
#[test_case(&[4096, 4096], 4096, true; "large gated")]
#[test_case(&[704, 2048], 512, false; "small m")]
#[test_case(&[300, 48], 200, false; "bk 16")]
#[test_case(&[1000, 96], 64, true; "bk 32 only, gated")]
fn gemm_candidates_are_valid(x: &[usize], n: usize, gated: bool) {
    let k = *x.last().unwrap();
    for smem in [sm86().smem_bytes, 48 << 10] {
        let target = Target { smem_bytes: smem, ..sm86() };
        let list = kernel_plan(shape::linear(Some(&planner(target)), &[BF16; 2], Some(&ext(x)), n, gated));
        assert!(!list.is_empty(), "{list:?}");
        for (i, c) in list.iter().enumerate() {
            assert!(k.is_multiple_of(c.tile[2]) && c.smem_bytes(gated) <= smem, "{c:?}");
            assert!(!list[..i].contains(c), "duplicate {c:?}");
        }
    }
}

/// A target capped at static shared memory drops a stage for the gated GEMM:
/// the three-slot ring of the 128×128 tile's 384-wide fill is 72 KB.
#[test]
fn gated_linear_drops_a_stage_under_a_static_cap() {
    let target = Target { smem_bytes: 48 << 10, ..sm86() };
    let planner = planner(target.clone());
    let list = kernel_plan(shape::linear(Some(&planner), &[BF16, BF16], Some(&ext(&[4096, 4096])), 4096, true));
    assert_valid(&list, &target, 4096, true);
    assert!(!list.iter().any(|c| c.tile == [128, 128, 32] && c.stages == 3), "{list:?}");
}

#[test_case(None, &[BF16, BF16], true, Fallback::Target; "no target")]
#[test_case(Some(apple()), &[BF16, BF16], true, Fallback::Target; "no tables for the arch")]
#[test_case(Some(sm86()), &[DType::Float32, DType::Float32], true, Fallback::Dtype; "f32 keeps the graph")]
#[test_case(Some(sm86()), &[BF16, DType::Float32], true, Fallback::Dtype; "mixed operand types")]
#[test_case(Some(sm86()), &[DType::Int32, DType::Int32], true, Fallback::Dtype; "integers")]
#[test_case(Some(sm86()), &[BF16, BF16], false, Fallback::Symbolic; "symbolic past the batch")]
fn every_op_falls_back_alike(target: Option<Target>, dtypes: &[DType], static_dims: bool, why: Fallback) {
    let x = static_dims.then(|| ext(&[2, 128, 1024]));
    let kv = static_dims.then(|| ext(&[2, 128, 16, 64]));
    let q = static_dims.then(|| ext(&[2, 128, 16, 64]));
    let planner = target.map(planner);
    let target = planner.as_ref();
    assert_eq!(shape::linear(target, dtypes, x.as_ref(), 1024, false), Plan::Graph(why));
    assert_eq!(shape::attention(target, dtypes, q.as_ref(), kv.as_ref(), false), Plan::Graph(why));
    assert_eq!(shape::norm(target, dtypes, x.as_ref()), Plan::Graph(why));
}

/// A bound variable that is the reduced dim itself is no batch.
#[test]
fn a_variable_reduced_dim_falls_back() {
    let x = Extent { dims: vec![64], var: true };
    assert_eq!(shape::linear(Some(&planner(sm86())), &[BF16; 2], Some(&x), 64, false), Plan::Graph(Fallback::Symbolic));
    let x = Extent { dims: vec![1024], var: true };
    assert_eq!(shape::norm(Some(&planner(sm86())), &[BF16; 2], Some(&x)), Plan::Graph(Fallback::Symbolic));
}

#[test]
fn f16_takes_the_kernels() {
    let f16 = [DType::Float16, DType::Float16];
    assert!(matches!(shape::linear(Some(&planner(sm86())), &f16, Some(&ext(&[64, 64])), 64, false), Plan::Kernel(_)));
}

/// The tune store keys a GEMM by its epilogue's text: without a scale it
/// reads as it did before the field, so stored choices still apply; a scale
/// is its own key and its own program.
#[test]
fn an_unscaled_epilogue_keeps_its_tune_key() {
    let plain = Epilogue { bias: true, residual: true, ..Epilogue::default() };
    assert_eq!(format!("{plain:?}"), "Epilogue { bias: true, act: None, gated: false, residual: true }");
    let half = Epilogue { scale: Some(Scale::new(0.5)), ..plain };
    assert_eq!(format!("{half:?}"), "Epilogue { bias: true, act: None, gated: false, residual: true, scale: 0.5 }");
    let spec = |epilogue| GemmSpec { m: 64, n: 64, k: 64, batch: Batch::Static(1), epilogue, cfg: BIG_MEASURED };
    assert_ne!(gemm::<BF16T>(&spec(plain)), gemm::<BF16T>(&spec(half)));
}

/// The graph fallback (f32 here) applies the scale between the activation and
/// the residual: `scale·act(x·wᵀ + b) + r`.
#[test_case(Act::None, false; "plain")]
#[test_case(Act::Silu, false; "silu")]
#[test_case(Act::Gelu, true; "geglu")]
fn the_graph_scales_before_the_residual(act: Act, gated: bool) {
    let f = |shape: &[usize], seed: usize| {
        let len = shape.iter().product::<usize>();
        let data: Vec<f32> = (0..len).map(|i| ((i * 7 + seed * 13) % 17) as f32 / 8.0 - 1.0).collect();
        Tensor::from_slice(data).try_reshape(shape.iter().map(|&d| d as isize).collect::<Vec<_>>()).unwrap()
    };
    let (m, n, k) = (5, 6, 4);
    let rows = if gated { 2 * n } else { n };
    let (x, w, b, r) = (f(&[m, k], 1), f(&[rows, k], 2), f(&[rows], 3), f(&[m, n], 4));
    let opts = Linear { bias: Some(&b), act, gated, residual: Some(&r), scale: Some(-0.75) };
    let got = tk::linear(&x, &w, opts).unwrap().to_vec::<f32>().unwrap();
    let unscaled = Linear { residual: None, scale: None, ..opts };
    let y = tk::linear(&x, &w, unscaled).unwrap().to_vec::<f32>().unwrap();
    let r = r.to_vec::<f32>().unwrap();
    for (i, g) in got.iter().enumerate() {
        let want = -0.75 * y[i] + r[i];
        assert!((g - want).abs() <= 1e-5 * want.abs().max(1.0), "[{i}] = {g}, want {want}");
    }
}

// ---- attention ---------------------------------------------------------------------

/// Every head size whose K/V fills divide among a block's threads plans
/// valid attention configs on sm_86 (a prefill of 100 queries over 37 keys),
/// with and without a bias. One whose layout count is past the register
/// file plans too: the compiled kernels decide, and the graph runs where
/// every one spills.
#[test_case(64, Ok(()); "d 64")]
#[test_case(128, Ok(()); "d 128")]
#[test_case(48, Ok(()); "d 48")]
#[test_case(32, Ok(()); "d 32")]
#[test_case(256, Ok(()); "d 256")]
#[test_case(40, Err(Fallback::Shape); "d 40 off the atoms")]
fn attention_plans(d: usize, want: Result<(), Fallback>) {
    let target = sm86();
    let (q, kv) = (ext(&[2, 100, 8, d]), ext(&[2, 37, 2, d]));
    for dtypes in [&[BF16; 3][..], &[BF16; 4]] {
        match shape::attention(Some(&planner(target.clone())), dtypes, Some(&q), Some(&kv), false) {
            Plan::Kernel(list) => {
                assert!(want.is_ok() && !list.is_empty(), "{list:?}");
                assert!(list.iter().all(|c| c.fits(&target, d) && c.smem_bytes(d) <= target.smem_bytes), "{list:?}");
            }
            Plan::Graph(why) => assert_eq!(Err(why), want),
        }
    }
}

/// A bias takes the kernel in the stream type only.
#[test_case(BF16, true; "bf16")]
#[test_case(DType::Float16, false; "f16 under bf16")]
#[test_case(DType::Float32, false; "f32")]
fn biased_attention_plans(bias: DType, kernel: bool) {
    let (q, kv) = (ext(&[2, 100, 8, 64]), ext(&[2, 100, 8, 64]));
    let plan = shape::attention(Some(&planner(sm86())), &[BF16, BF16, BF16, bias], Some(&q), Some(&kv), false);
    assert_eq!(matches!(plan, Plan::Kernel(_)), kernel, "{plan:?}");
    if !kernel {
        assert_eq!(plan, Plan::Graph(Fallback::Dtype));
    }
}

fn attention_list(d: usize, smem: usize) -> (Target, Arc<[FaCfg]>) {
    let target = Target { smem_bytes: smem, ..sm86() };
    let (q, kv) = (ext(&[2, 100, 8, d]), ext(&[2, 37, 2, d]));
    let list = kernel_plan(shape::attention(Some(&planner(target.clone())), &[BF16; 3], Some(&q), Some(&kv), false));
    (target, list)
}

/// The ranking spans query and key block sizes within shared memory, each
/// config once, its untested shortlist one pipeline per tile. The score
/// tile needs no scratch there: the `mma.sync` accumulator is its A operand.
#[test_case(64, 99 << 10; "d 64")]
#[test_case(128, 99 << 10; "d 128")]
#[test_case(128, 48 << 10; "d 128 under a static cap")]
fn attention_candidates(d: usize, smem: usize) {
    let (target, list) = attention_list(d, smem);
    assert!(list.len() >= 4, "{list:?}");
    assert!(list.iter().all(|c| c.smem_bytes(d) <= smem && c.fits(&target, d)), "{list:?}");
    for (i, c) in list.iter().enumerate() {
        assert!(!list[..i].contains(c), "{c:?} twice");
        assert_eq!(c.scratch_bytes(&target), 0);
    }
    let short = first_eight(&list);
    assert!(short.len() >= 4, "{short:?}");
    for (i, c) in short.iter().enumerate() {
        assert!(!short[..i].iter().any(|o| (o.bq, o.bkv) == (c.bq, c.bkv)), "one pipeline per tile: {short:?}");
    }
}

/// A warp past the queries adds the work of its padded rows. Where the
/// tiles fill the device, a decoder step's lead has one warp per 16
/// queries, no more; where they leave SIMDs idle, a padded warp runs on one
/// of them, so the lead has at most as many warps as an SM has SIMDs.
#[test_case(4, 1, 64; "one query, d 64, 32 tiles")]
#[test_case(4, 16, 128; "sixteen queries, d 128, 32 tiles")]
#[test_case(4, 17, 64; "seventeen queries, 32 tiles")]
#[test_case(4, 100, 64; "a prefill fills a block")]
#[test_case(64, 1, 64; "one query, d 64, 512 tiles")]
#[test_case(64, 16, 128; "sixteen queries, d 128, 512 tiles")]
fn decode_attention_leads_with_the_queries_warps(batch: usize, t: usize, d: usize) {
    let (q, kv) = (ext(&[batch, t, 8, d]), ext(&[batch, 1500, 8, d]));
    let list = kernel_plan(shape::attention(Some(&planner(sm86())), &[BF16; 3], Some(&q), Some(&kv), false));
    let (lead, queries) = (list[0], t.div_ceil(16) * 16);
    let filled = batch * 8 >= 16 * sm86().sms.unwrap() as usize;
    let most = if filled { queries } else { queries.max(16 * sm86().occupancy.simds as usize) };
    assert!(lead.bq <= most, "{lead:?} for {t} queries");
    if filled && t <= 16 {
        assert_eq!(lead.bq, 16, "{list:?}");
    }
}

/// One and two always; around two blocks per SM otherwise, never past the
/// key blocks.
#[test_case(20, 24, &[1, 2, 3, 6]; "twenty tiles on 28 SMs")]
#[test_case(20, 2, &[1, 2]; "two key blocks")]
#[test_case(224, 24, &[1, 2]; "enough tiles already")]
#[test_case(1, 24, &[1, 2, 24]; "one tile: the blocks cap")]
fn split_candidates(tiles: usize, blocks: usize, want: &[usize]) {
    assert_eq!(planner(sm86()).splits(tiles, blocks), want);
}

fn gfx1151() -> Target {
    Target { sms: Some(40), ..Target::for_arch(GpuArch::Amd(AmdArch::Gfx1151)) }
}

/// Under gfx1151's probed costs the probe shapes' rankings (B 4, H 8, T
/// 2048) hold the tiles measured fastest on that chip in their first eight:
/// `(64, 16)` at `d = 64`, and `(64, 16)` or `(128, 16)` at `d = 128`
/// (26 TFLOP/s, 2026-10-13).
#[test_case(64, false, &[(64, 16)]; "d 64")]
#[test_case(64, true, &[(64, 16)]; "d 64 causal")]
#[test_case(128, false, &[(64, 16), (128, 16)]; "d 128")]
#[test_case(128, true, &[(64, 16), (128, 16)]; "d 128 causal")]
fn gfx1151_attention_shortlists_hold_the_measured_tiles(d: usize, causal: bool, measured: &[(usize, usize)]) {
    let planner = Planner::new(Device { target: gfx1151(), costs: super::gfx1151_costs() });
    let list = first_eight(&planner.attention_ranked(32, 2048, 2048, d, causal));
    let tiles: Vec<(usize, usize)> = list.iter().map(|c| (c.bq, c.bkv)).collect();
    assert!(measured.iter().any(|m| tiles.contains(m)), "{measured:?} not in {tiles:?}");
}

/// Under gfx1151's probed costs no block of fewer warps than the CU's two
/// SIMDs leads the 4096³ ranking (one-warp 256×256 tiles led it, measured
/// at 1–3 TFLOP/s, before a round was bound by one wave's issue).
#[test]
fn gfx1151_gemm_ranking_leads_with_full_blocks() {
    let planner = Planner::new(Device { target: gfx1151(), costs: super::gfx1151_costs() });
    let list = first_eight(&planner.gemm_ranked(1, 4096, 4096, 4096, false));
    let simds = planner.device.target.occupancy.simds;
    assert!(list.iter().all(|c| c.warps[0] * c.warps[1] >= simds), "{list:?}");
}

/// A named split count is honoured: the configs with that many key blocks
/// stay, at that count; past the most any config has, it is capped there;
/// without one every config stays unsplit.
#[test_case(Some(2), 128, &[(16, 64, 2), (16, 32, 2)]; "two splits drop the one-block config")]
#[test_case(Some(4), 128, &[(16, 32, 4)]; "four splits keep the four-block config")]
#[test_case(Some(9), 128, &[(16, 32, 4)]; "past every config: capped at the most")]
#[test_case(Some(2), 16, &[(16, 128, 1), (16, 64, 1), (16, 32, 1)]; "one key block: unsplit")]
#[test_case(None, 128, &[(16, 128, 1), (16, 64, 1), (16, 32, 1)]; "none named: unsplit")]
fn a_named_split_count_is_honoured(splits: Option<usize>, tk: usize, want: &[(usize, usize, usize)]) {
    let ranked = [FaCfg::new(16, 128, 2), FaCfg::new(16, 64, 2), FaCfg::new(16, 32, 2)];
    let got: Vec<_> = split_ranked(&ranked, splits, tk).iter().map(|c| (c.bq, c.bkv, c.splits)).collect();
    assert_eq!(got, want);
}

/// The decoder step that lost its named splits on gfx1151 (sixteen queries
/// at `d = 48` over 128 cached keys, two splits): its whole ranking keeps
/// the two splits, whichever config leads.
#[test]
fn a_decoder_steps_ranking_keeps_its_named_splits() {
    let planner = Planner::new(Device { target: gfx1151(), costs: super::gfx1151_costs() });
    let ranked = planner.attention_ranked(5 * 16, 16, 128, 48, false);
    let split = split_ranked(&ranked, Some(2), 128);
    assert!(!split.is_empty() && split.iter().all(|c| c.splits == 2 && 128usize.div_ceil(c.bkv) >= 2), "{split:?}");
}

/// The softmax forms and the tiles are tuned in turn: a forms pass on the
/// ranking's first clean config, the tiles over the ranking under the
/// forms it kept, and a forms pass on their winner. A pass visits each form
/// once, in order, and keeps a flip only where it measured faster (a tie
/// keeps the forms so far), the store choosing as the tuner's does.
#[test]
fn the_forms_and_tiles_are_tuned_in_turn() {
    use crate::tune::{TuneKey, TuneStore};
    #[derive(Debug, PartialEq)]
    enum Asked {
        FirstClean,
        Pair(FaCfg, FaCfg),
        Tiles(Vec<FaCfg>),
    }
    let store = TuneStore::at(None);
    let ranked = [FaCfg::new(128, 32, 2), FaCfg::new(64, 16, 2)];
    // Truncation and the late sum measure faster, the raw maximum ties, and
    // under truncation the second tile wins.
    let ns = |c: &FaCfg| {
        let f = c.forms;
        100 - 10 * u64::from(f.truncate_p) - 5 * u64::from(f.sum_after) + u64::from(f.skip_rescale)
            - 20 * u64::from(f.truncate_p && c.bq == 64)
    };
    let asked = std::cell::RefCell::new(vec![]);
    let fastest = |list: &[FaCfg]| {
        let key = TuneKey::new("attention", &sm86(), 0, svod_dtype::ScalarDType::BFloat16, &[], &format!("{list:?}"));
        let chosen = store.select(&key, list.len(), || 0, || list.iter().map(|c| Some(ns(c))).collect());
        list[chosen.unwrap_or(0)]
    };
    let tuned = descend(
        &ranked,
        |r| {
            asked.borrow_mut().push(Asked::FirstClean);
            r[0]
        },
        |r| {
            asked.borrow_mut().push(Asked::Tiles(r.to_vec()));
            fastest(r)
        },
        |[current, flipped]| {
            asked.borrow_mut().push(Asked::Pair(current, flipped));
            fastest(&[current, flipped])
        },
    );
    let asked = asked.into_inner();
    assert_eq!(asked.len(), 10, "{asked:?}");
    assert_eq!(asked[0], Asked::FirstClean);
    let pairs = |at: std::ops::Range<usize>, tile: (usize, usize)| {
        for (k, a) in asked[at].iter().enumerate() {
            let Asked::Pair(current, flipped) = a else { panic!("a forms pair, not {a:?}") };
            assert_eq!((current.bq, current.bkv), tile);
            assert_eq!(flipped.forms, crate::ops::attention::FORMS[k](current.forms), "form {k} flipped");
        }
    };
    pairs(1..5, (128, 32));
    let forms = Forms { truncate_p: true, sum_after: true, ..Forms::default() };
    let formed = ranked.map(|c| FaCfg { forms, ..c }).to_vec();
    assert_eq!(asked[5], Asked::Tiles(formed), "the tiles under the first pass's forms");
    pairs(6..10, (64, 16));
    assert_eq!(tuned, FaCfg { forms, ..FaCfg::new(64, 16, 2) });
}

#[test]
fn attention_with_symbolic_keys_falls_back() {
    let q = ext(&[2, 100, 8, 64]);
    assert_eq!(
        shape::attention(Some(&planner(sm86())), &[BF16; 3], Some(&q), None, false),
        Plan::Graph(Fallback::Symbolic)
    );
}

// ---- heads -------------------------------------------------------------------------

#[test_case(64, Ok(NormCfg { br: 4 }); "d 64")]
#[test_case(128, Ok(NormCfg { br: 4 }); "d 128")]
#[test_case(48, Err(Fallback::Shape); "d 48")]
#[test_case(512, Err(Fallback::Shape); "d 512")]
fn heads_plan(d: usize, want: Result<NormCfg, Fallback>) {
    let x = ext(&[2, 100, 12 * d]);
    assert_eq!(first(shape::heads(Some(&planner(sm86())), &[BF16], Some(&x), d)), want);
}

#[test]
fn heads_keep_the_graph_for_a_weight_off_the_stream_dtype() {
    let x = ext(&[2, 100, 12 * 64]);
    assert_eq!(
        shape::heads(Some(&planner(sm86())), &[BF16, DType::Float16], Some(&x), 64),
        Plan::Graph(Fallback::Dtype)
    );
}

#[test]
fn heads_need_three_dims() {
    let x = ext(&[200, 12 * 64]);
    assert_eq!(shape::heads(Some(&planner(sm86())), &[BF16], Some(&x), 64), Plan::Graph(Fallback::Shape));
    assert_eq!(shape::heads(Some(&planner(sm86())), &[BF16], None, 64), Plan::Graph(Fallback::Symbolic));
}

// ---- norms -------------------------------------------------------------------------

#[test_case(1024, Ok(vec![4, 8, 16]); "1024")]
#[test_case(256, Ok(vec![4, 8, 16]); "256")]
#[test_case(768, Err(Fallback::Shape); "not a power of two")]
#[test_case(128, Err(Fallback::Shape); "narrower than a warp's loads")]
#[test_case(4096, Err(Fallback::Shape); "wider than the measured range")]
fn norm_plans(d: usize, want: Result<Vec<usize>, Fallback>) {
    let want = want.map_or_else(Plan::Graph, |rows| Plan::Kernel(rows.into_iter().map(|br| NormCfg { br }).collect()));
    assert_eq!(shape::norm(Some(&planner(sm86())), &[BF16; 2], Some(&ext(&[37, d]))), want);
}

// ---- extents -------------------------------------------------------------------------

/// Dims come back at capacity; only a bound leading variable is admitted.
#[test]
fn extents_take_a_bound_leading_variable_only() {
    let var = Variable::new("b", 1, 6);
    let bound = var.bind(2).unwrap();
    let t = Tensor::empty(&[6, 3, 5], BF16).try_shrink([Some((SInt::Const(0), bound.as_sint())), None, None]).unwrap();
    let (e, v) = extent(&t.shape().unwrap()).expect("a bound batch");
    assert_eq!(e, Extent { dims: vec![6, 3, 5], var: true });
    let v = v.unwrap();
    assert_eq!((v.name.as_str(), v.min, v.max, v.dim), ("b", 1, 6, bound.as_sint()));

    let (e, v) = extent(&Tensor::empty(&[6, 3], BF16).shape().unwrap()).unwrap();
    assert_eq!((e, v.is_none()), (Extent { dims: vec![6, 3], var: false }, true));

    let swapped = t.try_permute(&[1, 0, 2]).unwrap();
    assert!(extent(&swapped.shape().unwrap()).is_none(), "a symbolic non-leading dim");
}

#[test]
fn no_device_no_kernels() {
    assert!(!tk::supported(&DeviceSpec::Cpu));
}

// ---- errors ----------------------------------------------------------------------------

fn t(shape: &[usize], dtype: DType) -> Tensor {
    Tensor::empty(shape, dtype)
}

#[test]
fn semantic_mismatches_are_errors() {
    let x = t(&[4, 64], BF16);
    let err = |r: tk::Result<Tensor>| r.expect_err("an error");
    let err3 = |r: tk::Result<(Tensor, Tensor, Tensor)>| r.expect_err("an error");
    assert!(matches!(err(tk::linear(&x, &t(&[32, 48], BF16), Linear::default())), Error::Shape { operand: "w", .. }));
    assert!(matches!(
        err(tk::linear(&x, &t(&[32, 64], DType::Float16), Linear::default())),
        Error::Dtype { operand: "w", .. }
    ));
    let gated = Linear { gated: true, ..Linear::default() };
    assert!(matches!(err(tk::linear(&x, &t(&[33, 64], BF16), gated)), Error::Shape { operand: "w", .. }));
    let bias = t(&[32], BF16);
    let opts = Linear { bias: Some(&bias), gated: true, ..Linear::default() };
    assert!(matches!(err(tk::linear(&x, &t(&[64, 64], BF16), opts)), Error::Shape { operand: "bias", .. }));
    let residual = t(&[4, 64], BF16);
    let opts = Linear { residual: Some(&residual), ..Linear::default() };
    assert!(matches!(err(tk::linear(&x, &t(&[32, 64], BF16), opts)), Error::Shape { operand: "residual", .. }));

    let q = t(&[2, 10, 6, 64], BF16);
    let kv = t(&[2, 12, 4, 64], BF16);
    assert!(matches!(err(tk::attention(&q, &kv, &kv, Attn::default())), Error::Heads { heads: 6, kv_heads: 4, .. }));
    let kv2 = t(&[2, 12, 2, 64], BF16);
    let v = t(&[2, 11, 2, 64], BF16);
    assert!(matches!(err(tk::attention(&q, &kv2, &v, Attn::default())), Error::Shape { operand: "v", .. }));
    let short = t(&[2, 12, 2, 32], BF16);
    assert!(matches!(err(tk::attention(&q, &short, &short, Attn::default())), Error::Shape { operand: "k", .. }));
    let lens = t(&[2, 1], DType::Int32);
    let opts = Attn { keys: KeyMask::Lens(&lens), ..Attn::default() };
    assert!(matches!(err(tk::attention(&q, &kv2, &kv2, opts)), Error::Shape { operand: "key lens", .. }));
    let mask = t(&[2, 10], DType::Bool);
    let opts = Attn { keys: KeyMask::Bool(&mask), ..Attn::default() };
    assert!(matches!(err(tk::attention(&q, &kv2, &kv2, opts)), Error::Shape { operand: "key mask", .. }));
    let seg = t(&[2, 12], DType::Int32);
    let opts = Attn { seg_start: Some(&seg), ..Attn::default() };
    assert!(matches!(err(tk::attention(&q, &kv2, &kv2, opts)), Error::Shape { operand: "seg start", .. }));
    let cache = t(&[3, 12, 6, 64], BF16);
    let plain = Cache { head_start: 0, kv_heads: 2, row_map: None, appended: None, app_lens: None };
    let opts = Attn { cache: Some(plain), ..Attn::default() };
    assert!(
        matches!(err(tk::attention(&q, &cache, &cache, opts)), Error::Shape { operand: "k", .. }),
        "rows need a map"
    );
    let map = t(&[2], DType::Int32);
    let opts = Attn { cache: Some(Cache { head_start: 5, row_map: Some(&map), ..plain }), ..Attn::default() };
    assert!(
        matches!(err(tk::attention(&q, &cache, &cache, opts)), Error::Shape { operand: "k", .. }),
        "heads past the row"
    );
    let app = t(&[2, 1, 2, 64], BF16);
    let opts =
        Attn { cache: Some(Cache { row_map: Some(&map), appended: Some((&app, &app)), ..plain }), ..Attn::default() };
    assert!(
        matches!(err(tk::attention(&q, &cache, &cache, opts)), Error::Shape { operand: "key lens", .. }),
        "appended needs lens"
    );
    let lens = t(&[2], DType::Int32);
    let wide = t(&[2, 1, 3, 64], BF16);
    let with_lens = Attn { keys: KeyMask::Lens(&lens), ..Attn::default() };
    let opts = Attn { cache: Some(Cache { row_map: Some(&map), appended: Some((&app, &wide)), ..plain }), ..with_lens };
    assert!(matches!(err(tk::attention(&q, &cache, &cache, opts)), Error::Shape { operand: "appended v", .. }));
    let bias = t(&[2, 6, 10, 13], BF16);
    let opts = Attn { cache: Some(Cache { row_map: Some(&map), appended: Some((&app, &app)), ..plain }), ..with_lens };
    assert!(tk::attention(&q, &cache, &cache, Attn { bias: Some(&bias), ..opts }).is_ok(), "the appended column");
    for dims in [[2, 6, 10, 12], [3, 6, 10, 13], [2, 2, 10, 13], [2, 6, 12, 13]] {
        let bias = t(&dims, BF16);
        let opts = Attn { bias: Some(&bias), ..opts };
        let got = tk::attention(&q, &cache, &cache, opts);
        assert!(matches!(err(got), Error::Shape { operand: "bias", .. }), "{dims:?}");
    }
    let rows = t(&[2, 8, 2, 64], BF16);
    let eight = Cache { row_map: Some(&map), appended: Some((&rows, &rows)), ..plain };
    let wide = t(&[2, 6, 10, 20], BF16);
    let opts = Attn { cache: Some(eight), bias: Some(&wide), ..with_lens };
    assert!(tk::attention(&q, &cache, &cache, opts).is_ok(), "eight appended columns");
    let opts = Attn { cache: Some(Cache { app_lens: Some(&lens), ..eight }), ..with_lens };
    assert!(tk::attention(&q, &cache, &cache, opts).is_ok(), "counted appended rows");
    let opts = Attn { cache: Some(Cache { app_lens: Some(&map), row_map: Some(&map), ..plain }), ..with_lens };
    assert!(
        matches!(err(tk::attention(&q, &cache, &cache, opts)), Error::Shape { operand: "appended lens", .. }),
        "a count needs appended rows"
    );
    let opts = Attn { cache: Some(Cache { app_lens: Some(&seg), ..eight }), ..with_lens };
    assert!(matches!(err(tk::attention(&q, &cache, &cache, opts)), Error::Shape { operand: "appended lens", .. }));
    let none = t(&[2, 0, 2, 64], BF16);
    let opts = Attn { cache: Some(Cache { appended: Some((&none, &none)), ..eight }), ..with_lens };
    assert!(matches!(err(tk::attention(&q, &cache, &cache, opts)), Error::Shape { operand: "appended k", .. }));
    let bias = t(&[1, 6, 10, 12], BF16);
    assert!(tk::attention(&q, &kv2, &kv2, Attn { bias: Some(&bias), ..Attn::default() }).is_ok(), "shared");
    // Off the stream dtype is the graph's to handle, not an error.
    let bias = t(&[2, 6, 10, 12], DType::Float32);
    assert!(tk::attention(&q, &kv2, &kv2, Attn { bias: Some(&bias), ..Attn::default() }).is_ok(), "f32 bias");

    let qkv = t(&[2, 10, 8 * 64], BF16);
    let split = Qkv { heads: 4, kv_heads: 2, head_dim: 64, q_norm: None, k_norm: None, eps: 1e-6, rope: None };
    let bad = Qkv { kv_heads: 3, ..split };
    assert!(matches!(err3(tk::heads(&qkv, bad)), Error::Heads { heads: 4, kv_heads: 3, .. }));
    let bad = Qkv { heads: 8, ..split };
    assert!(matches!(err3(tk::heads(&qkv, bad)), Error::Shape { operand: "qkv", .. }));
    let w = t(&[32], BF16);
    let bad = Qkv { k_norm: Some(&w), ..split };
    assert!(matches!(err3(tk::heads(&qkv, bad)), Error::Shape { operand: "k norm", .. }));
    // Off the stream dtype is the graph's to handle, not an error.
    let w16 = t(&[64], DType::Float16);
    assert!(tk::heads(&qkv, Qkv { q_norm: Some(&w16), ..split }).is_ok());
    let cos = t(&[1, 10, 1, 64], BF16);
    let bad = Qkv { rope: Some((&cos, &cos)), ..split };
    assert!(matches!(err3(tk::heads(&qkv, bad)), Error::Shape { operand: "cos", .. }));
    let cos = t(&[1, 10, 1, 32], BF16);
    let sin = t(&[2, 10, 1, 32], BF16);
    let bad = Qkv { rope: Some((&cos, &sin)), ..split };
    assert!(matches!(err3(tk::heads(&qkv, bad)), Error::Shape { operand: "sin", .. }));

    let w = t(&[32], BF16);
    assert!(matches!(err(tk::layer_norm(&x, &w, None, 1e-5)), Error::Shape { operand: "w", .. }));
    let w = t(&[64], BF16);
    let r = t(&[4, 32], BF16);
    assert!(matches!(tk::add_rms_norm(&x, &r, &w, 1e-5).map(|_| ()), Err(Error::Shape { operand: "residual", .. })));
}

// ---- conv2d ------------------------------------------------------------------------

fn conv_geom(hw: usize, cin: usize, cout: usize, k: usize, s: usize) -> crate::kernels::conv::ConvGeom {
    crate::kernels::conv::ConvGeom {
        h: hw,
        w: hw,
        cin,
        cout,
        kernel: [k, k],
        stride: [s, s],
        pad: [k / 2, k / 2],
        dilation: [1, 1],
    }
}

/// The untuned pick's output width: never padding the channels by more than
/// an eighth (a body of 96 channels takes no 128-wide tile); a 48-channel
/// body, whose steps would be one depth, takes the graph on the cp.async path.
fn conv_plan(x: &[usize], g: crate::kernels::conv::ConvGeom, groups: usize) -> Result<usize, Fallback> {
    let plan = shape::conv2d(Some(&planner(sm86())), &[F16; 3], F16, None, Some(&ext(x)), &g, groups);
    first(plan).map(|c| c.gemm.tile[1])
}

const F16: DType = DType::Float16;

#[test_case(&[1, 80, 80, 384], conv_geom(80, 384, 96, 3, 1), 1, Ok(96); "head conv, 96 wide")]
#[test_case(&[1, 20, 20, 192], conv_geom(20, 192, 192, 3, 1), 1, Ok(192); "a starved shallow body")]
#[test_case(&[8, 80, 80, 384], conv_geom(80, 384, 384, 3, 2), 1, Ok(384); "a static batch folds into the rows")]
#[test_case(&[1, 40, 40, 192], conv_geom(40, 192, 192, 3, 1), 1, Ok(192); "a shallow body filling the device")]
#[test_case(&[1, 80, 80, 96], conv_geom(80, 96, 96, 3, 1), 1, Ok(96); "96 wide, shallow")]
#[test_case(&[1, 640, 640, 3], conv_geom(640, 3, 96, 3, 2), 1, Err(Fallback::Shape); "rgb stem")]
#[test_case(&[1, 160, 160, 48], conv_geom(160, 48, 48, 3, 1), 1, Err(Fallback::Config); "48 wide takes the graph")]
#[test_case(&[1, 20, 20, 64], conv_geom(20, 64, 60, 3, 1), 1, Err(Fallback::Shape); "cout off 8")]
#[test_case(&[1, 20, 20, 64], conv_geom(20, 64, 64, 3, 1), 64, Err(Fallback::Shape); "depthwise")]
#[test_case(&[0, 20, 20, 64], conv_geom(20, 64, 64, 3, 1), 1, Err(Fallback::Shape); "no images")]
#[test_case(&[1, 1, 1, 64], conv_geom(1, 64, 64, 5, 1).with_pad(0), 1, Err(Fallback::Shape); "empty output")]
fn conv_plans(x: &[usize], g: crate::kernels::conv::ConvGeom, groups: usize, want: Result<usize, Fallback>) {
    match (conv_plan(x, g, groups), want) {
        (Ok(bn), Ok(cout)) => {
            assert!(8 * (cout.div_ceil(bn) * bn - cout) <= cout.div_ceil(bn) * bn, "{bn} pads {cout}")
        }
        (got, want) => assert_eq!(got, want),
    }
}

#[test_case(&[F16, F16, F16], DType::Float32, Some(DType::Float32), true; "f32 out with an f32 residual")]
#[test_case(&[F16, F16, F16], DType::Float32, Some(F16), false; "residual off the output type")]
#[test_case(&[F16, F16, F16], DType::BFloat16, None, false; "another half output")]
#[test_case(&[DType::Float32; 3], DType::Float32, None, false; "f32 keeps the graph")]
#[test_case(&[F16, BF16, F16], F16, None, false; "mixed operands")]
fn conv_dtype_plans(dtypes: &[DType], out: DType, residual: Option<DType>, kernel: bool) {
    let g = conv_geom(80, 384, 96, 3, 1);
    let plan = shape::conv2d(Some(&planner(sm86())), dtypes, out, residual, Some(&ext(&[1, 80, 80, 384])), &g, 1);
    assert_eq!(matches!(plan, Plan::Kernel(_)), kernel, "{plan:?}");
    if !kernel {
        assert_eq!(plan, Plan::Graph(Fallback::Dtype));
    }
}

/// A bound batch walks grid z: candidates for one image's rows, the batch
/// counted toward the grid.
#[test]
fn a_bound_conv_batch_plans_per_image() {
    let g = conv_geom(80, 384, 384, 3, 2);
    let x = Extent { dims: vec![8, 80, 80, 384], var: true };
    let plan = shape::conv2d(Some(&planner(sm86())), &[F16; 3], F16, None, Some(&x), &g, 1);
    assert_eq!(plan, Plan::Kernel(planner(sm86()).conv_ranked(8, 1600, &g)));
    let plan = shape::conv2d(Some(&planner(apple())), &[F16; 3], F16, None, Some(&x), &g, 1);
    assert_eq!(plan, Plan::Graph(Fallback::Target));
    assert_eq!(
        shape::conv2d(Some(&planner(sm86())), &[F16; 3], F16, None, None, &g, 1),
        Plan::Graph(Fallback::Symbolic)
    );
}

/// RDNA plans kernels from the lattice under its own facts: two-slot
/// register-staged rings only, unrolled over the slots, 16×16 WMMA-tileable
/// warp grids, attention within 64 KB of LDS with its score relayout.
#[test]
fn rdna_plans_register_staged_kernels() {
    let target = rdna3();
    let gemms =
        kernel_plan(shape::linear(Some(&planner(target.clone())), &[BF16; 2], Some(&ext(&[4096, 4096])), 4096, false));
    assert_valid(&gemms, &target, 4096, false);
    assert!(first_eight(&gemms).iter().all(|c| c.stages == 2 && c.unroll && c.fits(&target)), "{gemms:?}");
    for (d, t) in [(64, 1500), (64, 1)] {
        let q = ext(&[1, t, 8, d]);
        let fa = kernel_plan(shape::attention(Some(&planner(target.clone())), &[BF16; 3], Some(&q), Some(&q), false));
        let within = |c: &FaCfg| c.smem_bytes(d) + c.scratch_bytes(&target) <= 64 << 10;
        assert!(fa.iter().all(|c| c.stages == 2 && within(c)), "{fa:?}");
        assert!(fa.iter().all(|c| c.scratch_bytes(&target) == c.bq * c.bkv * 4), "RDNA3 re-holds its scores");
    }
    assert!(matches!(
        shape::norm(Some(&planner(target.clone())), &[BF16; 2], Some(&ext(&[37, 1024]))),
        Plan::Kernel(_)
    ));
}
