//! The GEMM raster ([`TileOrder`]): every block takes a distinct tile and
//! every tile is taken, whatever the group height over ragged grids; a GEMM
//! walking any raster computes its product.

use proptest::prelude::*;
use svod_dtype::ScalarDType;
use test_case::test_case;

use crate::build::*;
use crate::interp::{round_to, run};
use crate::ir::Shape;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, TileOrder, gemm};
use crate::ops::config::{Tiled, gemm_cfg};
use crate::sweep::assumed;

/// The launch index (grid x fastest) of the block that takes each tile of a
/// `gm × gn` grid walked in groups of `rows`, row-major; `-1` where none.
fn walked(gm: usize, gn: usize, rows: usize) -> Vec<i64> {
    let order = TileOrder::grouped(rows).expect("a group of rows");
    let prog = Kernel::record("raster", |k| {
        let out = k.output::<I32>("tiles", gm * gn);
        k.grid([Sc::from(gm), Sc::from(gn), Sc::from(1)]);
        k.warps(1);
        let (row, col) = order.tile(k, gm, gn);
        let id = k.block(0) + k.block(1) * gm;
        let one = Shape::new(1, 1);
        let at = k.view(out, 0, [gn, 1], one, [None, None]);
        let at = k.at(at, row, col);
        let id = k.splat::<I32>(one, id);
        k.store(at, id);
    });
    let out = run(&prog, vec![vec![-1.0; gm * gn]], &[]).expect("the raster runs").remove(0);
    out.into_iter().map(|id| id as i64).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Every block takes a tile of its own and every tile is taken: a
    /// bijection of the launch onto the grid, for groups of one row to past
    /// the grid's rows over grids whose last group is short.
    #[test]
    fn every_block_takes_a_distinct_tile(
        (gm, gn, rows) in (1usize..=12, 1usize..=12).prop_flat_map(|(gm, gn)| (Just(gm), Just(gn), 1..=gm + 1)),
    ) {
        let mut ids = walked(gm, gn, rows);
        ids.sort_unstable();
        prop_assert_eq!(ids, (0..(gm * gn) as i64).collect::<Vec<_>>());
    }

    /// A config's code holds its raster whole: any group height up to the
    /// tallest round-trips, and a taller one (a hand-made code's) reads
    /// back as the tallest.
    #[test]
    fn a_raster_round_trips_its_code(rows in 1usize.., stages in 1usize..=3) {
        let order = TileOrder::grouped(rows).expect("a group of rows");
        prop_assert_eq!(order.rows(), rows.min(TileOrder::MOST_ROWS));
        let cfg = GemmCfg { order, ..gemm_cfg([128, 64, 32], stages, [2, 2], false) };
        let code = cfg.code();
        prop_assert!(code.ends_with(&format!(",g{},~", order.rows())), "{}", code);
        prop_assert_eq!(GemmCfg::of_code(&code), Some(cfg));
        let hand = code.replace(&format!(",g{},", order.rows()), &format!(",g{rows},"));
        prop_assert_eq!(GemmCfg::of_code(&hand), Some(cfg));
    }
}

/// One row a group walks the grid row-major, the grid's rows or more
/// column-major; three rows over seven walk two groups of three and one of
/// a row, each down its rows a column at a time.
#[test_case(5, 4, 1, |r, c| r * 4 + c; "row-major")]
#[test_case(5, 4, 5, |r, c| c * 5 + r; "column-major")]
#[test_case(5, 4, 9, |r, c| c * 5 + r; "past the rows")]
#[test_case(7, 2, 3, |r, c| if r < 6 { r / 3 * 6 + c * 3 + r % 3 } else { 12 + c }; "a short last group")]
fn the_raster_walks_its_groups(gm: usize, gn: usize, rows: usize, id: fn(usize, usize) -> usize) {
    let want: Vec<i64> = (0..gm * gn).map(|t| id(t / gn, t % gn) as i64).collect();
    assert_eq!(walked(gm, gn, rows), want);
}

/// The GEMM computes `a · bᵀ` whatever raster its blocks walk: a grid of
/// seven tile rows by three columns, rows and columns past the operands;
/// a group past the tallest (a hand-made code's) is clamped, its span of
/// tiles representable.
#[test_case(1; "row-major")]
#[test_case(3; "groups of three")]
#[test_case(8; "the default")]
#[test_case(7; "column-major")]
#[test_case(usize::MAX; "past the tallest")]
fn a_gemm_walking_any_raster_computes_its_product(rows: usize) {
    let (m, n, k) = (200, 150, 64);
    let order = TileOrder::grouped(rows).expect("a group of rows");
    let cfg = GemmCfg { order, ..gemm_cfg([32, 64, 32], 2, [1, 2], false) };
    let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg: assumed(cfg) };
    let mut seed = 11;
    let a = super::rounded(ScalarDType::BFloat16, m * k, &mut seed, |x| x);
    let b = super::rounded(ScalarDType::BFloat16, n * k, &mut seed, |x| x);
    let out = run(&gemm::<BF16>(&spec), vec![a.clone(), b.clone(), vec![0.0; m * n]], &[]).unwrap().remove(2);
    for i in 0..m {
        for j in 0..n {
            let want = round_to(ScalarDType::BFloat16, (0..k).map(|kk| a[i * k + kk] * b[j * k + kk]).sum());
            let got = out[i * n + j];
            assert!((got - want).abs() <= 2e-2 * want.abs().max(1.0), "c[{i},{j}] = {got}, want {want}");
        }
    }
}
