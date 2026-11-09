//! The tail split ([`TailPlan`]): the whole waves' run and the tail's run
//! cover every tile of a raster's walk once; the three programs compute the
//! plain GEMM under its epilogue in the interpreter; the planner splits a
//! partial last wave where the model prices it cheaper, never where the
//! tiles fill whole waves or dwarf one, at counts that divide the trips;
//! the config's code holds the split.

use proptest::prelude::*;
use svod_dtype::ScalarDType;
use test_case::test_case;

use crate::atoms::{Target, sm86};
use crate::build::*;
use crate::interp::{round_to, run};
use crate::ir::Shape;
use crate::kernels::gemm::{
    Epilogue, GemmCfg, GemmSpec, Run, Scale, TailPlan, TailSpec, TailSplit, TileOrder, Tiles, gemm, gemm_tail,
};
use crate::kernels::{Act, Batch};
use crate::ops::config::{GemmProblem, Planner, TAIL_SPLITS, Tiled, gemm_cfg};
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

/// What the planner ranks first on a shape: the config, its cost and its
/// plan.
fn lead(planner: &Planner, [m, n, k]: [usize; 3]) -> (GemmCfg, f64, Option<TailPlan>) {
    let p = GemmProblem { batches: 1, m, n, k, halves: 1, rope: None };
    let lead = planner.gemm_ranked(1, m, n, k, false)[0];
    (lead, p.admit(planner, &lead).expect("ranked").0, p.tail_plan(planner, &lead))
}

/// On a 304-CU card 8192×4096 at its 256² tiles is 512 tiles, a wave and
/// 208: the planner runs a wave data-parallel and splits the 208 tiles'
/// reduction, priced under every unsplit config's; its unsplit twin is
/// priced dearer still by more than the tuner resolves, so none is offered.
#[test]
fn a_partial_last_wave_is_split() {
    let planner = mi300x();
    let [m, n, k] = [8192, 4096, 4096];
    let (cfg, cost, plan) = lead(&planner, [m, n, k]);
    let plan = plan.unwrap_or_else(|| panic!("{cfg:?} splits no tail"));
    let p = GemmProblem { batches: 1, m, n, k, halves: 1, rope: None };
    let wave = p.wave(&planner, &cfg).expect("a wave");
    assert_eq!(cfg.tile[..2], [256, 256], "{cfg:?}");
    assert_eq!((wave, plan.waves.map(Run::count), plan.tail), (304, Some(304), Run::new(304, 208).unwrap()));
    let ranked = planner.gemm_ranked(1, m, n, k, false);
    let unsplit = ranked.iter().filter(|c| c.tail.is_none()).map(|c| p.admit(&planner, c).expect("ranked").0);
    assert!(unsplit.fold(f64::INFINITY, f64::min) > cost, "an unsplit config is priced under the split lead");
    let whole = p.admit(&planner, &GemmCfg { tail: None, ..cfg }).expect("its unsplit form").0;
    assert!(whole > cost * crate::tune::SETTLED, "{whole} against {cost}: within the tuner's resolution");
}

/// The planner runs every tile data-parallel where they fill whole waves
/// (4864×4096 at 256² tiles is 304 tiles, a wave) or dwarf one (65536×8192
/// is 27 waves less 96 tiles: a split tail's partials and merge cost more
/// than its idle share saves).
#[test_case([4864, 4096, 4096]; "a whole wave")]
#[test_case([65536, 8192, 1024]; "many waves")]
fn whole_waves_run_unsplit(shape: [usize; 3]) {
    let planner = mi300x();
    let (cfg, _, plan) = lead(&planner, shape);
    assert!(cfg.tail.is_none() && plan.is_none(), "{shape:?}: {cfg:?} splits {plan:?}");
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
    let p = GemmProblem { batches: 1, m, n, k, halves: 1, rope: None };
    let ranked = planner.gemm_ranked(1, m, n, k, false);
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
        let ranked = planner.gemm_ranked(batches, m, n, k, gated);
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
    let p = GemmProblem { batches: 1, m: 8192, n: 4096, k: 4096, halves: 1, rope: None };
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

/// Where a split leads, the tuner times the unsplit config beside it: the
/// split's model was never fitted to a measured time, and on small shapes
/// it ranks a whole split-K far ahead (37×96×48 on sm_86 is one tile).
#[test_case(sm86(), [37, 96, 48]; "one tile on sm_86")]
#[test_case(sm86(), [704, 512, 512]; "a partial wave on sm_86")]
#[test_case(Target::card("mi300x").expect("a card"), [128, 4096, 1024]; "a short wide gemm on mi300x")]
#[test_case(Target::card("mi300x").expect("a card"), [8192, 4096, 4096]; "a partial last wave on mi300x")]
fn a_leading_split_has_an_unsplit_twin(target: Target, [m, n, k]: [usize; 3]) {
    let planner = planner(target);
    let ranked = planner.gemm_ranked(1, m, n, k, false);
    let lead = ranked[0];
    assert!(lead.tail.is_some(), "{lead:?} leads unsplit");
    let twins = lead.twins(&ranked);
    assert!(twins.iter().any(|c| c.tail.is_none()), "{lead:?} has no unsplit twin in {twins:?}");
}
