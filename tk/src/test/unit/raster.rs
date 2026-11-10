//! The GEMM raster ([`TileOrder`]): the blocks walk groups of tile rows a
//! column at a time, every block a distinct tile and every tile taken,
//! whatever the group height over ragged grids; a GEMM walking any raster
//! computes its product.

use proptest::prelude::*;
use svod_dtype::ScalarDType;
use test_case::test_case;

use crate::build::*;
use crate::interp::{round_to, run};
use crate::ir::Shape;
use crate::kernels::gemm::{GemmCfg, TileOrder, Tiles, gemm};
use crate::ops::config::gemm_cfg;
use crate::sweep::assumed;

/// The walk index each tile of a `gm × gn` grid takes under `tiles` of a
/// raster in groups of `rows`, row-major; `-1` where none. Over the whole
/// grid a block's walk index is its launch index (grid x fastest).
pub(super) fn walked(gm: usize, gn: usize, rows: usize, tiles: Tiles) -> Vec<i64> {
    let order = TileOrder::grouped(rows).expect("a group of rows");
    let prog = Kernel::record("walk", |k| {
        let out = k.output::<I32>("tiles", gm * gn);
        let [gx, gy] = tiles.grid(gm, gn);
        k.grid([gx, gy, Sc::from(1)]);
        k.warps(1);
        let (row, col) = tiles.tile(k, order, gm, gn);
        let id = match tiles {
            Tiles::Grid => k.block(0) + k.block(1) * gm,
            Tiles::Run(run) => k.block(0) + run.first(),
        };
        let one = Shape::new(1, 1);
        let at = k.view(out, 0, [gn, 1], one, [None, None]);
        let at = k.at(at, row, col);
        let id = k.splat::<I32>(one, id);
        k.store(at, id);
    });
    let out = run(&prog, vec![vec![-1.0; gm * gn]], &[]).expect("the walk runs").remove(0);
    out.into_iter().map(|id| id as i64).collect()
}

/// The walk counted out by hand: groups of `rows` tile rows from the top,
/// the last one short, each down its rows a column at a time.
fn counted(gm: usize, gn: usize, rows: usize) -> Vec<i64> {
    let mut want = vec![-1; gm * gn];
    let mut id = 0;
    for top in (0..gm).step_by(rows) {
        for col in 0..gn {
            for row in top..(top + rows).min(gm) {
                want[row * gn + col] = id;
                id += 1;
            }
        }
    }
    want
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// The blocks walk the grid as counted out by hand, for groups of one
    /// row (row-major) to past the grid's rows (column-major) over grids
    /// whose last group is short: every block takes a tile of its own and
    /// every tile is taken.
    #[test]
    fn the_raster_walks_its_groups(
        (gm, gn, rows) in (1usize..=12, 1usize..=12).prop_flat_map(|(gm, gn)| (Just(gm), Just(gn), 1..=gm + 1)),
    ) {
        let mut ids = walked(gm, gn, rows, Tiles::Grid);
        prop_assert_eq!(&ids, &counted(gm, gn, rows));
        ids.sort_unstable();
        prop_assert_eq!(ids, (0..(gm * gn) as i64).collect::<Vec<_>>());
    }
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
    let spec = assumed(super::gemm_spec(m, n, k), cfg);
    let mut seed = 11;
    let a = super::rounded(ScalarDType::BFloat16, m * k, &mut seed, |x| x);
    let b = super::rounded(ScalarDType::BFloat16, n * k, &mut seed, |x| x);
    let out = run(&gemm::<BF16>(&spec), vec![a.clone(), b.clone(), vec![0.0; m * n]], &[]).unwrap().remove(2);
    let want: Vec<f64> =
        super::interp::naive_gemm(&a, &b, n, k).into_iter().map(|c| round_to(ScalarDType::BFloat16, c)).collect();
    super::assert_rel_close("c", &out, &want, 2e-2);
}
