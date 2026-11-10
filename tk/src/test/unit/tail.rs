//! The tail split ([`TailPlan`]): the whole waves' run and the tail's run
//! cover every tile of a raster's walk once; the three programs compute the
//! plain GEMM under its epilogue in the interpreter; the planner offers a
//! partial last wave's split where the model prices it cheaper, never where
//! the tiles fill whole waves or dwarf one, at counts that divide the
//! trips, and only as a twin the tuner times beside its unsplit pipeline;
//! the config's code holds the split.

use proptest::prelude::*;
use svod_dtype::ScalarDType;
use test_case::test_case;

use super::{costs, gfx942_costs, gfx1201_costs, sm86_costs};
use crate::atoms::{Costs, Device};
use crate::atoms::{Target, sm86};
use crate::build::*;
use crate::interp::{round_to, run};
use crate::ir::Shape;
use crate::kernels::gemm::{
    Epilogue, GemmCfg, GemmSpec, Run, Scale, TailPlan, TailSpec, TailSplit, TileOrder, Tiles, gemm, gemm_tail,
};
use crate::kernels::{Act, Batch};
use crate::ops::config::{GemmProblem, LAUNCH, Planner, TAIL_SPLITS, Tiled, gemm_cfg};
use crate::sweep::{assumed, planner};

/// The walk index each tile of a `gm × gn` grid takes under `run` of a
/// raster in groups of `rows`, row-major; `-1` where the run takes none.
fn covered(gm: usize, gn: usize, rows: usize, run_: Run) -> Vec<i64> {
    let (order, tiles) = (TileOrder::grouped(rows).expect("a group of rows"), Tiles::Run(run_));
    let prog = Kernel::record("run", |k| {
        let out = k.output::<I32>("tiles", gm * gn);
        let [gx, gy] = tiles.grid(gm, gn);
        k.grid([gx, gy, Sc::from(1)]);
        k.warps(1);
        let (row, col) = tiles.tile(k, order, gm, gn);
        let id = k.block(0) + run_.first();
        let one = Shape::new(1, 1);
        let at = k.view(out, 0, [gn, 1], one, [None, None]);
        let at = k.at(at, row, col);
        let id = k.splat::<I32>(one, id);
        k.store(at, id);
    });
    let out = run(&prog, vec![vec![-1.0; gm * gn]], &[]).expect("the run walks").remove(0);
    out.into_iter().map(|id| id as i64).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// The whole waves' run and the tail's run take every tile of the grid
    /// exactly once between them, the waves a whole number of waves from
    /// the walk's first and the tail short of one, whatever the raster,
    /// over grids whose last group is short; tiles that fill whole waves
    /// leave no tail.
    #[test]
    fn the_waves_and_the_tail_take_every_tile_once(
        (gm, gn, rows, wave) in (1usize..=12, 1usize..=12)
            .prop_flat_map(|(gm, gn)| (Just(gm), Just(gn), 1..=gm + 1, 1..=gm * gn + 3)),
        split in 2usize..=TAIL_SPLITS,
    ) {
        let tiles = gm * gn;
        let split = TailSplit::new(split).expect("a split");
        let Some(plan) = TailPlan::new(tiles, wave, split) else {
            prop_assert!(tiles.is_multiple_of(wave), "{} tiles over waves of {} leave a tail", tiles, wave);
            return Ok(());
        };
        prop_assert_eq!(plan.split, split.split());
        prop_assert!(plan.tail.count() < wave && plan.tail.range().end == tiles, "{:?}", plan);
        prop_assert_eq!(plan.waves.map_or(0, |w| w.count()), plan.tail.first());
        prop_assert!(plan.waves.is_none_or(|w| w.first() == 0 && w.count().is_multiple_of(wave)), "{:?}", plan);
        let mut taken = vec![0usize; tiles];
        let mut ids = vec![];
        for r in plan.waves.into_iter().chain([plan.tail]) {
            for (tile, id) in covered(gm, gn, rows, r).into_iter().enumerate() {
                if id >= 0 {
                    prop_assert!(r.range().contains(&(id as usize)), "{:?} took walk index {}", r, id);
                    taken[tile] += 1;
                    ids.push(id);
                }
            }
        }
        prop_assert!(taken.iter().all(|&n| n == 1), "tiles taken {:?} times", taken);
        ids.sort_unstable();
        prop_assert_eq!(ids, (0..tiles as i64).collect::<Vec<_>>());
    }

    /// A config's code holds its tail split whole, the tile drops it, and
    /// no split under two reads back.
    #[test]
    fn a_tail_split_round_trips_its_code(split in 0usize..=64, rows in 1usize..=16) {
        let order = TileOrder::grouped(rows).expect("a group of rows");
        let base = GemmCfg { order, ..gemm_cfg([128, 128, 64], 2, [2, 2], true) };
        let unsplit = base.code();
        let code = format!("{},t{split}", unsplit.strip_suffix(",~").expect("an unsplit code"));
        match TailSplit::new(split) {
            Some(tail) => {
                let cfg = GemmCfg { tail: Some(tail), ..base };
                prop_assert_eq!(cfg.code(), code.clone());
                prop_assert_eq!(GemmCfg::of_code(&code), Some(cfg));
                prop_assert_eq!(cfg.tile(), base.tile());
            }
            None => prop_assert_eq!(GemmCfg::of_code(&code), None),
        }
        prop_assert_eq!(GemmCfg::of_code(&unsplit), Some(base));
    }
}

/// `x` activated as the kernel activates it: gelu by Abramowitz–Stegun
/// 7.1.26's erf.
fn activated(act: Act, x: f64) -> f64 {
    match act {
        Act::None => x,
        Act::Silu => x / (1.0 + (-x).exp()),
        Act::Gelu => {
            let z = x / std::f64::consts::SQRT_2;
            let t = 1.0 / (1.0 + 0.3275911 * z.abs());
            let poly =
                t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
            let erf = (1.0 - poly * (-z * z).exp()).copysign(z);
            0.5 * x * (1.0 + erf)
        }
    }
}

const BIAS: Epilogue = Epilogue { bias: true, ..Epilogue::DEFAULT };
const GELU_RESIDUAL: Epilogue = Epilogue { bias: true, act: Act::Gelu, residual: true, ..Epilogue::DEFAULT };
const SILU_SCALE: Epilogue = Epilogue { bias: true, act: Act::Silu, scale: Some(Scale::new(0.5)), ..Epilogue::DEFAULT };
const F32_RESIDUAL: Epilogue = Epilogue { residual: true, out_f32: true, ..Epilogue::DEFAULT };

/// A tail split computes what the plain GEMM computes, and both the naive
/// product under the epilogue, over ragged grids (rows and columns past the
/// operands) whose tiles take waves of a small device and a tail: the
/// waves' GEMM into the output, the tail's partials, their merge into the
/// rest, which the waves' tiles keep; a grid short of a wave splits whole.
#[test_case(200, 160, 256, [64, 64, 64], [2, 2], 3, 5, 2, GELU_RESIDUAL; "groups of three, waves of five")]
#[test_case(130, 96, 128, [32, 64, 32], [1, 2], 1, 4, 4, Epilogue::DEFAULT; "row-major, a four-way tail")]
#[test_case(100, 200, 192, [64, 64, 32], [2, 2], 8, 7, 3, SILU_SCALE; "the default raster, a tail of one")]
#[test_case(96, 280, 64, [32, 128, 32], [1, 2], 3, 4, 2, BIAS; "merge tiles past the columns of a tile")]
#[test_case(70, 100, 128, [32, 64, 32], [1, 2], 3, 64, 2, F32_RESIDUAL; "short of a wave, an f32 output")]
#[test_case(260, 64, 128, [64, 64, 32], [2, 2], 5, 3, 4, BIAS; "column-major")]
#[allow(clippy::too_many_arguments)]
fn a_tail_split_computes_the_gemm(
    m: usize,
    n: usize,
    k: usize,
    tile: [usize; 3],
    warps: [u32; 2],
    rows: usize,
    wave: usize,
    split: usize,
    epilogue: Epilogue,
) {
    let split = TailSplit::new(split).expect("a split");
    let order = TileOrder::grouped(rows).expect("a group of rows");
    let cfg = GemmCfg { order, tail: Some(split), ..gemm_cfg(tile, 2, warps, false) };
    assert!(TailSpec::merges(&cfg), "{cfg:?}: the merge tile divides the GEMM's");
    let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue, cfg: assumed(cfg) };
    let plan = TailPlan::new(m.div_ceil(tile[0]) * n.div_ceil(tile[1]), wave, split).expect("a tail");
    let tail = TailSpec { gemm: spec.clone(), plan };
    let out = if epilogue.out_f32 { ScalarDType::Float32 } else { ScalarDType::BFloat16 };
    let mut seed = 23;
    let mut draw = |dtype, len| super::rounded(dtype, len, &mut seed, |x| x);
    let (a, b) = (draw(ScalarDType::BFloat16, m * k), draw(ScalarDType::BFloat16, n * k));
    let bias = epilogue.bias.then(|| draw(ScalarDType::BFloat16, n));
    let residual = epilogue.residual.then(|| draw(out, m * n));
    let with = |front: Vec<Vec<f64>>, c: Vec<f64>| -> Vec<Vec<f64>> {
        front.into_iter().chain(bias.clone()).chain(residual.clone()).chain([c]).collect()
    };
    let last = |mut outs: Vec<Vec<f64>>| outs.pop().expect("the output");
    let operands = || vec![a.clone(), b.clone()];

    let plain = last(run(&gemm::<BF16>(&spec), with(operands(), vec![0.0; m * n]), &[]).unwrap());
    let mut programs = gemm_tail::<BF16>(&tail).into_iter();
    assert_eq!(programs.len(), 2 + usize::from(plan.waves.is_some()), "{plan:?}");
    // NaN where nothing is stored: a tile the merge leaves would read so.
    let mut c = vec![f64::NAN; m * n];
    if plan.waves.is_some() {
        c = last(run(&programs.next().expect("the waves"), with(operands(), c), &[]).unwrap());
    }
    let partials = programs.next().expect("the partials");
    let mut parts = operands();
    parts.push(vec![f64::NAN; tail.partials()]);
    let parts = last(run(&partials, parts, &[]).unwrap());
    // A slot holds its tile's elements inside the output and nothing past
    // it, where a device's accumulator holds no defined value.
    let ([bm, bn, _], gn) = (tile, n.div_ceil(tile[1]));
    let walk = covered(m.div_ceil(bm), gn, rows, plan.tail);
    for (e, v) in parts.iter().enumerate() {
        let (slot, r, col) = (e / (bm * bn) % plan.tail.count(), e / bn % bm, e % bn);
        let at = walk.iter().position(|&w| w == (plan.tail.first() + slot) as i64).expect("a tail tile");
        let (tr, tc) = (at / gn, at % gn);
        let inside = tr * bm + r < m && tc * bn + col < n;
        assert_eq!(v.is_nan(), !inside, "{plan:?}: partial {e} of tile ({tr}, {tc}) at ({r}, {col})");
    }
    let got = last(run(&programs.next().expect("the merge"), with(vec![parts], c), &[]).unwrap());

    for i in 0..m {
        for j in 0..n {
            let dot: f64 = (0..k).map(|p| a[i * k + p] * b[j * k + p]).sum();
            let mut want = activated(epilogue.act, dot + bias.as_ref().map_or(0.0, |b| b[j]));
            want *= epilogue.scale.map_or(1.0, |s| f64::from(s.get()));
            want = round_to(out, want + residual.as_ref().map_or(0.0, |r| r[i * n + j]));
            let (g, p) = (got[i * n + j], plain[i * n + j]);
            let tol = 2e-2 * want.abs().max(1.0);
            assert!((g - want).abs() <= tol, "{plan:?}: c[{i},{j}] = {g}, want {want}");
            assert!((g - p).abs() <= tol, "{plan:?}: c[{i},{j}] = {g}, the plain GEMM's {p}");
        }
    }
}

/// The card's planner under the declared (guessed) costs: a 304-CU MI300X.
fn mi300x() -> Planner {
    planner(Target::card("mi300x").expect("a listed card"))
}

/// What the planner ranks first on a shape and its tile's split twin
/// (`None` where it offers none), each with its cost and plan.
#[allow(clippy::type_complexity)]
fn lead(planner: &Planner, [m, n, k]: [usize; 3]) -> ((GemmCfg, f64), Option<(GemmCfg, f64, TailPlan)>) {
    let p = GemmProblem::new(1, m, n, k, &Epilogue::DEFAULT);
    let ranked = planner.gemm_ranked(1, m, n, k, &Epilogue::DEFAULT);
    let lead = ranked[0];
    let split = ranked.iter().find(|c| c.tail.is_some() && c.tile() == lead.tile()).map(|&c| {
        let plan = p.tail_plan(planner, &c).unwrap_or_else(|| panic!("{c:?} plans no tail"));
        (c, p.admit(planner, &c).expect("ranked").0, plan)
    });
    ((lead, p.admit(planner, &lead).expect("ranked").0), split)
}

/// On a 304-CU card 8192×4096 at its 256² tiles is 512 tiles, a wave and
/// 208: the model prices the plan that runs a wave data-parallel and
/// splits the 208 tiles' reduction under every unsplit config, by more
/// than the tuner resolves, yet the ranking leads with the tile unsplit
/// and offers the split as its twin, for the tuner to time against it.
#[test]
fn a_partial_last_wave_is_split() {
    let planner = mi300x();
    let [m, n, k] = [8192, 4096, 4096];
    let ((lead, cost), split) = lead(&planner, [m, n, k]);
    let (cfg, split_cost, plan) = split.unwrap_or_else(|| panic!("{lead:?} has no split twin"));
    let p = GemmProblem::new(1, m, n, k, &Epilogue::DEFAULT);
    let wave = p.wave(&planner, &cfg).expect("a wave");
    assert!(lead.tail.is_none(), "{lead:?} leads split");
    assert_eq!(lead.tile[..2], [256, 256], "{lead:?}");
    assert_eq!((wave, plan.waves.map(Run::count), plan.tail), (304, Some(304), Run::new(304, 208).unwrap()));
    let ranked = planner.gemm_ranked(1, m, n, k, &Epilogue::DEFAULT);
    let unsplit = ranked.iter().filter(|c| c.tail.is_none()).map(|c| p.admit(&planner, c).expect("ranked").0);
    assert_eq!(unsplit.fold(f64::INFINITY, f64::min), cost, "the lead is the cheapest unsplit config");
    let whole = p.admit(&planner, &GemmCfg { tail: None, ..cfg }).expect("its unsplit form").0;
    assert!(whole > split_cost * crate::tune::SETTLED, "{whole} against {split_cost}: within the tuner's resolution");
}

/// Where the tiles fill whole waves (4864×4096 at 256² tiles is 304
/// tiles, a wave) no split of the leading tile runs, and where they dwarf
/// one (65536×8192 is 27 waves less 96 tiles) its split's partials and
/// merge cost more than its idle share saves.
#[test_case([4864, 4096, 4096]; "a whole wave")]
#[test_case([65536, 8192, 1024]; "many waves")]
fn whole_waves_run_unsplit(shape: [usize; 3]) {
    let planner = mi300x();
    let ((lead, cost), split) = lead(&planner, shape);
    let p = GemmProblem::new(1, shape[0], shape[1], shape[2], &Epilogue::DEFAULT);
    assert!(lead.tail.is_none() && p.tail_plan(&planner, &lead).is_none(), "{shape:?}: {lead:?} splits");
    if let Some((cfg, split_cost, plan)) = split {
        assert!(split_cost > cost, "{shape:?}: {cfg:?} {plan:?} priced at {split_cost} under {cost}");
    }
}

/// Every split the planner ranks, on any card and shape, divides its
/// tile's trips, is at most [`TAIL_SPLITS`], runs a single plain GEMM and
/// leaves a tail of a partial wave behind whole ones; its tile's whole
/// waves leave a tail, and the tile is ranked unsplit wherever they do not.
#[test_case(Target::card("mi300x").unwrap(), [8192, 4096, 4096])]
#[test_case(Target::card("mi300x").unwrap(), [6144, 4096, 8192])]
#[test_case(Target::card("mi300x").unwrap(), [4096, 4096, 4096])]
#[test_case(Target::card("mi300x").unwrap(), [128, 4096, 4096])]
#[test_case(sm86(), [704, 1536, 512])]
#[test_case(sm86(), [4096, 4096, 4096])]
#[test_case(Target::card("rx9070xt").unwrap(), [1500, 1280, 5120])]
#[test_case(Target::card("h100-sxm").unwrap(), [8192, 8192, 1024])]
fn every_ranked_split_divides_its_trips(target: Target, [m, n, k]: [usize; 3]) {
    let planner = planner(target);
    let p = GemmProblem::new(1, m, n, k, &Epilogue::DEFAULT);
    let ranked = planner.gemm_ranked(1, m, n, k, &Epilogue::DEFAULT);
    for cfg in ranked.iter() {
        let tiles = m.div_ceil(cfg.tile[0]) * n.div_ceil(cfg.tile[1]);
        let wave = p.wave(&planner, cfg).expect("ranked");
        let Some(tail) = cfg.tail else { continue };
        let (trips, split) = (k / cfg.tile[2], tail.split());
        assert!(trips.is_multiple_of(split) && split <= TAIL_SPLITS, "{cfg:?}: {trips} trips");
        let plan = p.tail_plan(&planner, cfg).unwrap_or_else(|| panic!("{cfg:?} plans no tail"));
        assert_eq!(Some(plan), TailPlan::new(tiles, wave, tail), "{cfg:?}");
        assert!(!tiles.is_multiple_of(wave), "{cfg:?}: {tiles} tiles fill waves of {wave}");
    }
    for (batches, gated) in [(2, false), (1, true)] {
        let ranked = planner.gemm_ranked(batches, m, n, k, &Epilogue { gated, ..Epilogue::DEFAULT });
        assert!(ranked.iter().all(|c| c.tail.is_none()), "a batched or gated GEMM splits");
    }
}

/// A split is refused where its kernels cannot run the problem: a count
/// that does not divide the trips, a batched, gated or turned GEMM, tiles
/// that fill whole waves, a convolution.
#[test]
fn a_split_the_problem_cannot_run_is_refused() {
    use crate::ops::config::{Rejected, ShapeRejected};
    let planner = mi300x();
    let refused = Err(Rejected::Shape(ShapeRejected::Tail));
    let cfg = gemm_cfg([128, 128, 64], 1, [4, 2], false);
    let split = |s| GemmCfg { tail: TailSplit::new(s), ..cfg };
    let p = GemmProblem::new(1, 8192, 4096, 4096, &Epilogue::DEFAULT);
    assert!(p.admit(&planner, &split(4)).is_ok());
    assert_eq!(p.admit(&planner, &split(3)).map(|_| ()), refused, "64 trips split three ways");
    let rope = Some(crate::kernels::gemm::Rope { heads: 4, head_dim: 64, seq: 8 });
    for other in [GemmProblem { batches: 2, ..p }, GemmProblem { halves: 2, ..p }, GemmProblem { rope, ..p }] {
        assert_eq!(other.admit(&planner, &split(4)).map(|_| ()), refused, "{other:?}");
    }
    // 38 tile rows by 32: two waves of 608 (two blocks a CU).
    let whole = GemmProblem { m: 4864, ..p };
    assert_eq!(whole.wave(&planner, &cfg), Some(608));
    assert_eq!(whole.admit(&planner, &split(4)).map(|_| ()), refused, "whole waves");
}

/// A tail split's f32 partials take at most an eighth of the device's
/// memory, the share a score matrix's block takes: on a card with exactly
/// eight times its partials' bytes the split runs, a byte less and it is
/// refused with what it needed, and the ranking holds no split whose
/// partials pass the bound.
#[test]
fn a_split_whose_partials_outgrow_the_memory_is_refused() {
    use crate::ops::config::{Rejected, ShapeRejected};
    let cfg = GemmCfg { tail: TailSplit::new(4), ..gemm_cfg([128, 128, 64], 1, [4, 2], false) };
    let p = GemmProblem::new(1, 8192, 4096, 4096, &Epilogue::DEFAULT);
    let on = |memory: Option<u64>| {
        let target = Target { memory_bytes: memory, ..Target::card("mi300x").expect("a listed card") };
        Planner::new(Device { target, costs: costs() }).expect("an SM count")
    };
    let plan = p.tail_plan(&on(None), &cfg).expect("a tail");
    let need = plan.split * plan.tail.count() * 128 * 128 * 4;
    assert!(p.admit(&on(Some(8 * need as u64)), &cfg).is_ok());
    let most = need - 4;
    let refused = Err(Rejected::Shape(ShapeRejected::Partials { need, most }));
    assert_eq!(p.admit(&on(Some(8 * need as u64 - 1)), &cfg).map(|_| ()), refused);
    let planner = on(Some(8 * need as u64 - 1));
    for split in planner.gemm_ranked(1, 8192, 4096, 4096, &Epilogue::DEFAULT).iter().filter(|c| c.tail.is_some()) {
        let plan = p.tail_plan(&planner, split).expect("a ranked split plans its tail");
        let bytes = plan.split * plan.tail.count() * split.tile[0] * split.tile[1] * 4;
        assert!(bytes < need, "{split:?} holds {bytes} bytes of partials");
    }
}

/// A block's output moves its type's bytes once its trips are done, with
/// its residual's (of the output's type) and its bias columns' (both
/// halves' where gated); a tail's merge stores and reads the same a row and
/// its merge tile's bias once, beside the partials: between a plain
/// epilogue and an f32 output over a residual and a bias, a tail split's
/// price moves by exactly those bytes at DRAM's rate, its waves' blocks
/// one after another on the busiest SM and its merge's rows over the SMs.
#[test]
fn an_output_moves_its_types_bytes_and_its_epilogues_reads() {
    let wide = Epilogue { bias: true, residual: true, out_f32: true, ..Epilogue::DEFAULT };
    let gated = Epilogue { bias: true, gated: true, ..Epilogue::DEFAULT };
    let [m, n, k] = [8192, 4096, 4096];
    let mut planner = mi300x();
    planner.weights.epilogue = 0.0;
    let unsplit = gemm_cfg([128, 128, 64], 1, [4, 2], false);
    let out = |e: &Epilogue| planner.claim_of(&GemmProblem::new(1, m, n, k, e), &unsplit).expect("admitted").out;
    assert_eq!(out(&Epilogue::DEFAULT), (128 * 128 * 2) as f64);
    assert_eq!(out(&wide), (128 * 128 * 8 + 128 * 2) as f64);
    assert_eq!(out(&gated), (128 * 128 * 2 + 128 * 4) as f64);
    let cfg = GemmCfg { tail: TailSplit::new(4), ..unsplit };
    let (plain, wide) = (GemmProblem::new(1, m, n, k, &Epilogue::DEFAULT), GemmProblem::new(1, m, n, k, &wide));
    let plan = plain.tail_plan(&planner, &cfg).expect("a tail");
    assert_eq!(wide.tail_plan(&planner, &cfg), Some(plan));
    let waves = plan.waves.expect("whole waves").count().div_ceil(planner.sms());
    let rows = plan.tail.count() * 128;
    let (block, row) = (128 * 128 * 6 + 128 * 2, 128 * 6 + 128 * 2 / 32);
    let global = planner.device.costs.global_byte;
    let want = (waves * block) as f64 * global + (rows * row) as f64 * global / planner.sms() as f64;
    let moved = wide.plan_price(&planner, &cfg, &plan).expect("priced")
        - plain.plan_price(&planner, &cfg, &plan).expect("priced");
    assert!((moved / want - 1.0).abs() < 1e-9, "{moved} against {want}");
}

/// Where the model prices a split cheapest, the ranking still leads
/// unsplit and holds the split only as a twin of its unsplit pipeline: the
/// split's model is fitted to a few probed shapes, and it ran 4–8 %
/// optimistic on eleven MI300X shapes. Neither the untuned first pick
/// (`SVOD_TK_TUNE=0`, or a store not yet written: the first clean
/// representative, `first_clean` over `representatives`, whatever the
/// compiler refuses or spills) nor the shortlist takes a split, a whole
/// split-K of a grid short of a wave among them; the tuner's twins stage
/// times the split beside its tile's pipelines. Under the guessed costs
/// and under the cards' measured ones: on sm_86 4096³ splits a tail of
/// eight tiles of 512 (the device ran it 2.3 % faster), on MI300X
/// 6144×4096×8192 a tail of its last partial wave (19 % faster at 256²),
/// and 128×4096×1024 splits whole.
#[test_case(Target::card("mi300x").expect("a card"), costs(), [128, 4096, 1024]; "a short wide gemm on mi300x")]
#[test_case(Target::card("mi300x").expect("a card"), costs(), [8192, 4096, 4096]; "a partial last wave on mi300x")]
#[test_case(sm86(), sm86_costs(), [4096, 4096, 4096]; "a short tail on a measured sm_86")]
#[test_case(Target::card("mi300x").expect("a card"), gfx942_costs(), [6144, 4096, 8192]; "a measured mi300x")]
fn a_split_runs_only_as_a_timed_twin(target: Target, costs: Costs, [m, n, k]: [usize; 3]) {
    use crate::ops::config::{first_clean, representatives};
    use crate::tune::Compiled;
    let planner = Planner::new(Device { target, costs }).expect("an SM count");
    let p = GemmProblem::new(1, m, n, k, &Epilogue::DEFAULT);
    let ranked = planner.gemm_ranked(1, m, n, k, &Epilogue::DEFAULT);
    let price = |c: &GemmCfg| p.admit(&planner, c).expect("ranked").0;
    let cheapest = ranked.iter().copied().min_by(|a, b| price(a).total_cmp(&price(b))).expect("a ranking");
    assert!(cheapest.tail.is_some(), "{cheapest:?}: the model prices no split cheapest");
    assert!(ranked[0].tail.is_none(), "{:?} leads split", ranked[0]);
    let reps = representatives(&ranked);
    assert!(reps.iter().all(|c| c.tail.is_none()), "a split stands for its tile");
    // The untuned pick: every compile clean, the first tiles refused, every
    // one spilling.
    let verdicts: [&dyn Fn(usize) -> Compiled; 3] =
        [&|_| Compiled::Clean, &|at| if at < 3 { Compiled::Failed } else { Compiled::Clean }, &|_| Compiled::Spilled];
    for verdict in verdicts {
        let mut at = 0;
        let first = first_clean(&reps, |batch| {
            at += batch.len();
            (at - batch.len()..at).map(verdict).collect()
        });
        let first = reps[first.expect("a config compiles")];
        assert!(first.tail.is_none(), "the untuned pick {first:?} splits");
    }
    // The twins stage of the cheapest split's tile times it beside the
    // tile's unsplit pipelines.
    let leader = reps.iter().find(|c| c.tile() == cheapest.tile()).expect("the split's tile is represented");
    let twins = leader.twins(&ranked);
    assert!(twins.contains(&cheapest), "{cheapest:?} is no twin of {leader:?}: {twins:?}");
    assert!(twins.iter().any(|c| c.tail.is_none()), "{leader:?} has no unsplit twin in {twins:?}");
    for split in ranked.iter().filter(|c| c.tail.is_some()) {
        let pipeline = GemmCfg { tail: None, ..*split };
        let at = ranked.iter().position(|c| c == split).expect("ranked");
        assert_eq!(ranked[at - 1], pipeline, "{split:?} does not follow the pipeline it splits");
        assert!(price(split) <= price(&pipeline) * crate::tune::SETTLED, "{split:?} priced past its pipeline");
    }
}

/// A row of `gemm_tail_probe`: a shape at the config the planner led with
/// on the card (its tile, ring, warps and raster), and the device µs of its
/// data-parallel run, of a whole split-K and of a tail split, each at the
/// count the model priced cheapest then (each plan's kernels' times summed,
/// best of two passes in opposite orders, the plain epilogue). A tail
/// split of a grid under a wave is the whole split-K, timed once.
struct Probed {
    shape: [usize; 3],
    cfg: GemmCfg,
    dp: f64,
    whole: (usize, f64),
    tail: Option<(usize, f64)>,
}

#[allow(clippy::too_many_arguments)]
fn probed(
    shape: [usize; 3],
    tile: [usize; 3],
    stages: usize,
    warps: [u32; 2],
    rows: usize,
    dp: f64,
    whole: (usize, f64),
    tail: Option<(usize, f64)>,
) -> Probed {
    let order = TileOrder::grouped(rows).expect("a group of rows");
    Probed { shape, cfg: GemmCfg { order, ..gemm_cfg(tile, stages, warps, false) }, dp, whole, tail }
}

/// MI300X (304 CUs), 2026-11-09, under [`gfx942_costs`]' store.
fn mi300x_probed() -> Vec<Probed> {
    let probed = |shape, rows, dp, whole, tail| probed(shape, [256, 256, 64], 1, [4, 4], rows, dp, whole, tail);
    vec![
        probed([8192, 4096, 4096], 8, 443.0, (4, 710.8), Some((4, 520.0))),
        probed([4096, 4096, 4096], 8, 218.3, (2, 276.2), None),
        probed([8192, 8192, 1024], 16, 340.9, (2, 578.0), Some((2, 361.3))),
        probed([6144, 4096, 8192], 8, 1004.3, (2, 834.7), Some((8, 809.7))),
        probed([4096, 8192, 4096], 16, 475.8, (4, 737.2), Some((4, 553.0))),
        probed([12288, 4096, 4096], 1, 677.4, (4, 1111.4), Some((4, 721.4))),
        probed([8192, 6144, 2048], 8, 439.1, (2, 623.2), Some((2, 490.0))),
        probed([3072, 4096, 4096], 8, 217.6, (4, 292.0), None),
    ]
}

/// RTX 3060 (28 SMs), 2026-11-09.
fn sm86_probed() -> Vec<Probed> {
    let probed = |shape, tile, rows, dp, whole, tail| probed(shape, tile, 2, [4, 4], rows, dp, whole, tail);
    let (tall, wide) = ([256, 128, 64], [128, 256, 64]);
    vec![
        probed([8192, 4096, 4096], tall, 4, 12013.6, (2, 13061.1), Some((8, 11973.9))),
        probed([4096, 4096, 4096], tall, 4, 6166.2, (2, 6518.4), Some((8, 6022.4))),
        probed([8192, 8192, 1024], wide, 16, 6239.2, (2, 8582.1), Some((4, 6259.3))),
        probed([6144, 4096, 8192], tall, 4, 18471.9, (2, 18430.9), Some((2, 18251.7))),
    ]
}

/// RX 9070 XT (64 CUs), 2026-11-09: every grid whole waves, no tail.
fn gfx1201_probed() -> Vec<Probed> {
    let probed = |shape, rows, dp, whole| probed(shape, [256, 256, 64], 1, [4, 4], rows, dp, whole, None);
    vec![
        probed([8192, 4096, 4096], 8, 2535.4, (2, 3861.4)),
        probed([4096, 4096, 4096], 8, 1238.0, (2, 1909.9)),
        probed([8192, 8192, 1024], 8, 1208.7, (2, 4268.8)),
        probed([6144, 4096, 8192], 1, 4976.8, (2, 5588.1)),
    ]
}

/// Measurement: plans whose device times are this close are a tie the
/// probe does not resolve (its two passes drift by as much).
const TIE: f64 = 1.01;

/// The model orders a probed shape's plans — data-parallel, a whole
/// split-K, a tail split, each at its probed count — as the device did,
/// every pair the device told apart past [`TIE`], under the probe's own
/// plain epilogue and under none. Each
/// plan is priced as the probe timed it, its kernels' alone, without the
/// [`LAUNCH`]es between them.
#[test_case(Target::card("mi300x").expect("a card"), gfx942_costs(), mi300x_probed(); "mi300x")]
#[test_case(sm86(), sm86_costs(), sm86_probed(); "sm_86")]
#[test_case(Target::card("rx9070xt").expect("a card"), gfx1201_costs(), gfx1201_probed(); "gfx1201")]
fn the_model_orders_the_probed_plans_as_the_device_did(target: Target, costs: Costs, rows: Vec<Probed>) {
    for epilogue in [1.0, 0.0] {
        let mut planner = Planner::new(Device { target: target.clone(), costs }).expect("an SM count");
        planner.weights.epilogue = epilogue;
        for Probed { shape: [m, n, k], cfg, dp, whole, tail } in &rows {
            let p = GemmProblem::new(1, *m, *n, *k, &Epilogue::DEFAULT);
            let (tiles, wave) =
                (m.div_ceil(cfg.tile[0]) * n.div_ceil(cfg.tile[1]), p.wave(&planner, cfg).expect("a wave"));
            let kernels = |plan: TailPlan| {
                let launches = usize::from(plan.waves.is_some()) + 2;
                p.plan_price(&planner, cfg, &plan).expect("priced") - (launches - 1) as f64 * LAUNCH
            };
            let mut plans = vec![("data-parallel", *dp, p.admit(&planner, cfg).expect("admitted").0)];
            plans.push(("whole split-K", whole.1, kernels(TailPlan::whole(tiles, whole.0).expect("a plan"))));
            if let Some((split, us)) = *tail {
                let plan = TailPlan::new(tiles, wave, TailSplit::new(split).expect("a split")).expect("a tail");
                assert!(plan.waves.is_some(), "{m}x{n}x{k}: {plan:?} runs no whole wave");
                plans.push(("tail split", us, kernels(plan)));
            }
            let text: Vec<String> =
                plans.iter().map(|(what, us, ns)| format!("{what} {:.3}/{:.3}", us / dp, ns / plans[0].2)).collect();
            for (i, a) in plans.iter().enumerate() {
                for b in &plans[i + 1..] {
                    if a.1.max(b.1) <= a.1.min(b.1) * TIE {
                        continue;
                    }
                    assert_eq!(
                        a.1 < b.1,
                        a.2 < b.2,
                        "{m}x{n}x{k}, epilogue x{epilogue}: {} and {} (device/model x data-parallel: {text:?})",
                        a.0,
                        b.0
                    );
                }
            }
        }
    }
}

/// Where the device ran a probed shape's tile faster with its tail split
/// than data-parallel past [`TIE`] (sm_86 at 4096³, MI300X at
/// 6144×4096×8192), the ranking offers the tile's split for the tuner to
/// time; every probed pipeline leads its tile unsplit.
#[test_case(Target::card("mi300x").expect("a card"), gfx942_costs(), mi300x_probed(); "mi300x")]
#[test_case(sm86(), sm86_costs(), sm86_probed(); "sm_86")]
fn the_planner_offers_a_split_where_the_device_ran_one_faster(target: Target, costs: Costs, rows: Vec<Probed>) {
    let planner = Planner::new(Device { target, costs }).expect("an SM count");
    for Probed { shape: [m, n, k], cfg, dp, tail, .. } in &rows {
        let ranked = planner.gemm_ranked(1, *m, *n, *k, &Epilogue::DEFAULT);
        let pipeline = |c: &&GemmCfg| GemmCfg { order: cfg.order, tail: None, ..**c } == *cfg;
        let first = ranked.iter().find(pipeline).unwrap_or_else(|| panic!("{m}x{n}x{k}: {cfg:?} unranked"));
        assert!(first.tail.is_none(), "{m}x{n}x{k}: {first:?} leads its pipeline split");
        let Some((_, us)) = *tail else { continue };
        if us * TIE < *dp {
            let split = ranked.iter().find(|c| c.tail.is_some() && c.tile() == cfg.tile());
            assert!(split.is_some(), "{m}x{n}x{k}: no split of {cfg:?} to time, the device ran it faster");
        }
    }
}
