//! The nearest-candidate kernel: the interpreter against an f64 reference on
//! the host (every metric, floors, groups, ties, rows left with no
//! candidate), every ranked config against the reference on the device, the
//! op against its graph, and the throughput probe.

use proptest::prelude::*;
use svod_dtype::{AmdArch, DType, ScalarDType};
use svod_tensor::Tensor;
use test_case::test_case;

use super::first_eight;
use crate::atoms::{Core, Target};
use crate::build::BF16;
use crate::interp::{round_to, run};
use crate::ir::Orient;
use crate::kernels::nearest::{
    Floors, MergeSpec, Metric, NearestCfg, NearestSpec, nearest, nearest_merge, share_stride,
};
use crate::ops::config::Planner;
use crate::ops::{self as tk, Nearest};

/// A problem on the host: `x [n, d]` and `c [m, d]` exact in bf16.
#[derive(Clone, Debug)]
struct Case {
    n: usize,
    m: usize,
    d: usize,
    metric: Metric,
    x: Vec<f64>,
    c: Vec<f64>,
    row_floor: Option<Vec<f64>>,
    col_floor: Option<Vec<f64>>,
    groups: Option<(Vec<i32>, Vec<i32>)>,
}

/// How a case's values are drawn.
#[derive(Clone, Copy, Debug)]
struct Draw {
    metric: Metric,
    floors: Floors,
    /// Distinct group ids, `None` without groups; one id skips every pair.
    groups: Option<i32>,
    /// Small integers: every product and sum is exact, so ties are exact on
    /// every path and the reference's index is the only right one.
    integer: bool,
    /// Candidates past the third repeat the first three (with their floors
    /// and groups): exact ties whatever the values.
    repeated: bool,
}

const fn draw(metric: Metric, floors: Floors, groups: Option<i32>, integer: bool, repeated: bool) -> Draw {
    Draw { metric, floors, groups, integer, repeated }
}

const NONE: Floors = Floors { row: false, col: false };
const BOTH: Floors = Floors { row: true, col: true };

impl Case {
    fn draw(n: usize, m: usize, d: usize, draw: Draw, seed: u64) -> Self {
        let mut seed = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut value = |scale: f64| {
            let x = super::lcg(&mut seed);
            if draw.integer { (x * 2.5 * scale).round() } else { round_to(ScalarDType::BFloat16, x * scale) }
        };
        let x: Vec<f64> = (0..n * d).map(|_| value(1.0)).collect();
        let mut c: Vec<f64> = (0..m * d).map(|_| value(1.0)).collect();
        // Floors on either side of zero, some above the scores.
        let mut floor = |on: bool, rows: usize| on.then(|| (0..rows).map(|_| value(2.0) + 1.0).collect::<Vec<f64>>());
        let row_floor = floor(draw.floors.row, n);
        let mut col_floor = floor(draw.floors.col, m);
        let mut groups = draw.groups.map(|k| {
            let group =
                |rows: usize, salt: usize| (0..rows).map(|i| ((i * 7 + salt) % k as usize) as i32).collect::<Vec<_>>();
            (group(n, 0), group(m, 3))
        });
        if draw.repeated {
            for j in 3..m {
                let from = j % 3;
                c.copy_within(from * d..(from + 1) * d, j * d);
                if let Some(f) = &mut col_floor {
                    f[j] = f[from];
                }
                if let Some((_, g)) = &mut groups {
                    g[j] = g[from];
                }
            }
        }
        Case { n, m, d, metric: draw.metric, x, c, row_floor, col_floor, groups }
    }

    fn floors(&self) -> Floors {
        Floors { row: self.row_floor.is_some(), col: self.col_floor.is_some() }
    }

    fn row(v: &[f64], d: usize, i: usize) -> &[f64] {
        &v[i * d..(i + 1) * d]
    }

    /// Pair `(i, j)`'s score by the definition, `+∞` where its groups match.
    fn pair(&self, i: usize, j: usize) -> f64 {
        if self.groups.as_ref().is_some_and(|(r, c)| r[i] == c[j]) {
            return f64::INFINITY;
        }
        let (xi, cj) = (Self::row(&self.x, self.d, i), Self::row(&self.c, self.d, j));
        let dot: f64 = xi.iter().zip(cj).map(|(a, b)| a * b).sum();
        let sq = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>();
        let s = match self.metric {
            Metric::SquaredL2 => sq(xi) + sq(cj) - 2.0 * dot,
            Metric::ShiftedL2 => sq(cj) - 2.0 * dot,
            Metric::L2 => (sq(xi) + sq(cj) - 2.0 * dot).max(0.0).sqrt(),
            Metric::NegDot => -dot,
        };
        let row = self.row_floor.as_ref().map_or(f64::NEG_INFINITY, |f| f[i]);
        let col = self.col_floor.as_ref().map_or(f64::NEG_INFINITY, |f| f[j]);
        s.max(row).max(col)
    }

    /// Each row's least score and the smallest index scoring it; `(+∞, −1)`
    /// where every pair is skipped.
    fn reference(&self) -> (Vec<f64>, Vec<i64>) {
        (0..self.n)
            .map(|i| {
                (0..self.m).fold((f64::INFINITY, -1), |(best, at), j| {
                    let s = self.pair(i, j);
                    if s < best { (s, j as i64) } else { (best, at) }
                })
            })
            .unzip()
    }

    /// The first candidate holding the same row, floor and group as `j`.
    fn first_alike(&self, j: usize) -> usize {
        let same = |a: usize| {
            Self::row(&self.c, self.d, a) == Self::row(&self.c, self.d, j)
                && self.col_floor.as_ref().is_none_or(|f| f[a] == f[j])
                && self.groups.as_ref().is_none_or(|(_, g)| g[a] == g[j])
        };
        (0..=j).find(|&a| same(a)).expect("j itself")
    }

    /// The kernel's parameters at `depth` (zero columns past `d`), the norms
    /// in f32 as the op computes them, then the two outputs (of every share,
    /// split `splits` ways).
    fn params(&self, depth: usize, splits: usize) -> Vec<Vec<f64>> {
        let pad = |v: &[f64], rows: usize| {
            (0..rows * depth)
                .map(|e| if e % depth < self.d { v[e / depth * self.d + e % depth] } else { 0.0 })
                .collect()
        };
        let norms = |v: &[f64], rows: usize| -> Vec<f64> {
            (0..rows)
                .map(|i| round_to(ScalarDType::Float32, Self::row(v, self.d, i).iter().map(|a| a * a).sum()))
                .collect()
        };
        let mut params = vec![pad(&self.x, self.n), pad(&self.c, self.m)];
        if self.metric.col_norms() {
            params.push(norms(&self.c, self.m));
        }
        if self.metric.row_norms() {
            params.push(norms(&self.x, self.n));
        }
        params.extend(self.row_floor.iter().chain(&self.col_floor).cloned());
        if let Some((r, c)) = &self.groups {
            params.extend([r, c].map(|g| g.iter().map(|&v| f64::from(v)).collect()));
        }
        let parts = if splits == 1 { self.n } else { splits * share_stride(self.n) };
        params.extend([vec![0.0; parts], vec![0.0; parts]]);
        params
    }

    fn spec(&self, depth: usize, cfg: NearestCfg) -> NearestSpec {
        let (n, m, metric, floors, groups) = (self.n, self.m, self.metric, self.floors(), self.groups.is_some());
        NearestSpec { n, m, d: depth, metric, floors, groups, cfg }
    }

    /// The scores and indices the interpreter computes under `cfg`, split,
    /// its shares merged.
    fn interpreted(&self, cfg: NearestCfg) -> (Vec<f64>, Vec<f64>) {
        let depth = self.d.next_multiple_of(16);
        let mut out = run(&nearest::<BF16>(&self.spec(depth, cfg)), self.params(depth, cfg.splits), &[]).unwrap();
        let (index, score) = (out.pop().unwrap(), out.pop().unwrap());
        if cfg.splits == 1 {
            return (score, index);
        }
        merged(&MergeSpec { n: self.n, splits: cfg.splits, metric: self.metric }, score, index)
    }

    /// The scores and indices `cfg` computes on the device (split, its
    /// shares merged there).
    fn on_device(&self, planner: &Planner, depth: usize, cfg: NearestCfg) -> (Vec<f64>, Vec<f64>) {
        let prog = nearest::<BF16>(&self.spec(depth, cfg));
        let mut outs = super::targets::on_device(prog, &cfg.lowering(&planner.device), &self.params(depth, cfg.splits));
        let (index, score) = (outs.pop().flatten().unwrap(), outs.pop().flatten().unwrap());
        if cfg.splits == 1 {
            return (score, index);
        }
        let merge = MergeSpec { n: self.n, splits: cfg.splits, metric: self.metric };
        let params = [score, index, vec![0.0; self.n], vec![0.0; self.n]];
        let mut outs = super::targets::on_device(nearest_merge(&merge), &merge.lowering(&planner.device), &params);
        let index = outs.pop().flatten().unwrap();
        (outs.pop().flatten().unwrap(), index)
    }

    /// `score` and `index` are each row's least and where: every score
    /// within `tol` of the reference's (relative, absolute below one; L2's
    /// squared) and
    /// `+∞` exactly where it is; every index admissible, scoring the row's
    /// least within `tol` (a near-tie may fall either way under another
    /// rounding) and the first of the candidates alike; and, where `exact`,
    /// the reference's own.
    fn check(&self, what: &str, score: &[f64], index: &[f64], tol: f64, exact: bool) {
        let (best, at) = self.reference();
        assert_eq!((score.len(), index.len()), (self.n, self.n), "{what}: rows");
        // L2 is compared squared: a distance near zero is the root of a
        // difference of norms, its error that of the norms.
        let l2 = self.metric == Metric::L2;
        let near = |a: f64, b: f64| {
            let (a, b) = if l2 { (a * a, b * b) } else { (a, b) };
            a == b || (a - b).abs() <= tol * b.abs().max(1.0)
        };
        for i in 0..self.n {
            let (s, j) = (score[i], index[i]);
            assert!(near(s, best[i]), "{what}: row {i} scores {s}, want {}", best[i]);
            if best[i].is_infinite() {
                assert_eq!(j, -1.0, "{what}: row {i} has no candidate");
                continue;
            }
            assert!(j >= 0.0 && (j as usize) < self.m, "{what}: row {i} picks {j}");
            let j = j as usize;
            assert!(near(self.pair(i, j), best[i]), "{what}: row {i} picks {j} scoring {}", self.pair(i, j));
            assert_eq!(self.first_alike(j), j, "{what}: row {i} picks {j}, a repeat of an earlier candidate");
            if exact {
                assert_eq!(j as i64, at[i], "{what}: row {i}");
            }
        }
    }
}

/// The interpreter's merge of a split kernel's shares.
fn merged(spec: &MergeSpec, score: Vec<f64>, index: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let params = vec![score, index, vec![0.0; spec.n], vec![0.0; spec.n]];
    let mut out = run(&nearest_merge(spec), params, &[]).unwrap();
    let index = out.pop().unwrap();
    (out.pop().unwrap(), index)
}

/// The configs of a small lattice: every ring depth, one or two atom rows
/// per warp, the rows resident or streamed 16 or 32 deep a trip, the
/// candidates split up to four ways (at most a share per candidate tile).
fn small_cfg() -> impl Strategy<Value = NearestCfg> {
    let shape = (0usize..3, 0usize..3, 1usize..4, any::<bool>(), any::<bool>());
    (shape, prop::option::of(0usize..2), 1usize..5).prop_map(|((q, kv, stages, wide, unroll), bk, splits)| {
        let bq = 16 << q;
        let warp_rows = if wide && bq >= 32 { 32 } else { 16 };
        NearestCfg { unroll, warp_rows, bk: bk.map(|k| 16 << k), splits, ..NearestCfg::new(bq, 16 << kv, stages) }
    })
}

/// `cfg` with at most a share per candidate tile of `m` candidates.
fn shared_out(cfg: NearestCfg, m: usize) -> NearestCfg {
    NearestCfg { splits: cfg.splits.min(m.div_ceil(cfg.bkv)), ..cfg }
}

fn draws() -> impl Strategy<Value = Draw> {
    let metric = prop_oneof![Just(Metric::SquaredL2), Just(Metric::ShiftedL2), Just(Metric::L2), Just(Metric::NegDot)];
    let floors = (any::<bool>(), any::<bool>()).prop_map(|(row, col)| Floors { row, col });
    (metric, floors, prop::option::of(1i32..5), any::<bool>(), any::<bool>())
        .prop_map(|(metric, floors, groups, integer, repeated)| draw(metric, floors, groups, integer, repeated))
}

proptest! {
    /// The program computes each row's least score and its first index over
    /// any shape (ragged rows and candidates, depths padded to the core's,
    /// a streamed depth's last chunk overhanging it, the candidates split
    /// into shares the merge folds), every metric, either floor, with and
    /// without groups (rows left with no candidate included), and exact
    /// ties.
    #[test]
    fn nearest_interpreter_matches_the_reference(
        n in 1usize..70,
        m in 1usize..90,
        d in 1usize..40,
        draw in draws(),
        cfg in small_cfg(),
        seed in any::<u64>(),
    ) {
        let case = Case::draw(n, m, d, draw, seed);
        let (score, index) = case.interpreted(shared_out(cfg, m));
        case.check("interpreter", &score, &index, 1e-4, draw.integer);
    }
}

/// One group for every row and candidate skips every pair: each row scores
/// `+∞` at index −1; a candidate of another group is found again.
#[test]
fn rows_without_a_candidate_score_infinity_at_minus_one() {
    let mut case = Case::draw(40, 50, 16, draw(Metric::L2, BOTH, Some(1), false, false), 1);
    let (score, index) = case.interpreted(NearestCfg::new(32, 16, 2));
    assert!(score.iter().all(|s| *s == f64::INFINITY) && index.iter().all(|i| *i == -1.0));
    case.groups.as_mut().unwrap().1[37] = 2;
    let (score, index) = case.interpreted(NearestCfg::new(32, 16, 2));
    assert!(index.iter().all(|i| *i == 37.0), "{index:?}");
    case.check("one admissible candidate", &score, &index, 1e-4, false);
}

/// Equal scores in different trips, in one trip and in different shares
/// resolve to the smallest index: every candidate repeats one row, so each
/// row's least is everywhere (under the inner product, of either sign).
#[test_case(16, Metric::SquaredL2, 1; "a trip per pair of candidates")]
#[test_case(64, Metric::SquaredL2, 1; "every candidate in one trip")]
#[test_case(16, Metric::NegDot, 1; "signed scores a trip per pair of candidates")]
#[test_case(64, Metric::NegDot, 1; "signed scores in one trip")]
#[test_case(16, Metric::SquaredL2, 3; "three shares")]
#[test_case(16, Metric::NegDot, 4; "signed scores a share per trip")]
fn ties_resolve_to_the_smallest_index(bkv: usize, metric: Metric, splits: usize) {
    let mut case = Case::draw(30, 50, 8, draw(metric, NONE, None, true, false), 2);
    for j in 1..case.m {
        case.c.copy_within(0..case.d, j * case.d);
    }
    if metric == Metric::NegDot {
        let (best, _) = case.reference();
        assert!(best.iter().any(|s| *s < 0.0) && best.iter().any(|s| *s > 0.0), "scores of both signs: {best:?}");
    }
    let (score, index) = case.interpreted(NearestCfg { splits, ..NearestCfg::new(16, bkv, 2) });
    assert!(index.iter().all(|i| *i == 0.0), "{index:?}");
    case.check("ties", &score, &index, 0.0, true);
}

/// Where a row's least falls among the candidates: in the last one, each
/// candidate scoring below every earlier one (every trip lowers every row's
/// least); in the first one, the rest scoring above it (no trip after the
/// first lowers it); everywhere at once, every candidate alike.
#[derive(Clone, Copy, Debug)]
enum Order {
    Falling,
    Rising,
    Flat,
}

/// Rows `x_i = (1 + i mod 3, …)` against candidates `c_j = (v_j, …)` of small
/// integers in `order`, every other entry a small integer too: the inner
/// products are exact and their order is `order`'s whatever the row.
fn ordered(n: usize, m: usize, d: usize, order: Order) -> Case {
    let mut case = Case::draw(n, m, d, draw(Metric::NegDot, NONE, None, true, false), 5);
    for i in 0..n {
        case.x[i * d] = 1.0 + (i % 3) as f64;
        case.x[i * d + 1..(i + 1) * d].fill(0.0);
    }
    for j in 0..m {
        case.c[j * d] = match order {
            Order::Falling => (j + 1) as f64,
            Order::Rising => (m - j) as f64,
            Order::Flat => 7.0,
        };
    }
    case
}

/// The least is found and placed whichever trip lowers it: every trip, the
/// first alone, or the first with every later one tying it.
#[test_case(Order::Falling, 200; "the last candidate")]
#[test_case(Order::Rising, 0; "the first candidate")]
#[test_case(Order::Flat, 0; "every candidate alike")]
fn the_trip_that_lowers_a_row_places_its_least(order: Order, want: usize) {
    let case = ordered(40, 201, 16, order);
    for cfg in [NearestCfg::new(32, 16, 2), NearestCfg::new(16, 64, 1), NearestCfg::streamed(32, 32, 16, 2)] {
        let (score, index) = case.interpreted(cfg);
        assert!(index.iter().all(|&i| i == want as f64), "{cfg:?}: {index:?}");
        case.check(&format!("{cfg:?}"), &score, &index, 0.0, true);
    }
}

/// A floor above every score ties a row's candidates: the first one wins,
/// scoring the floor.
#[test]
fn a_floor_above_every_score_picks_the_first_candidate() {
    let mut case = Case::draw(20, 40, 16, draw(Metric::L2, Floors { row: true, col: false }, None, false, false), 3);
    case.row_floor = Some(vec![100.0; case.n]);
    let (score, index) = case.interpreted(NearestCfg::new(16, 16, 2));
    assert!(score.iter().all(|s| *s == 100.0) && index.iter().all(|i| *i == 0.0), "{score:?} {index:?}");
}

/// Split into shares, each row's least and first index are the unsplit
/// ones, bit for bit: exact ties run across every share boundary (each
/// candidate repeats one of the first three), the last candidate tile is
/// ragged, under every metric, floors and groups, the rows resident or
/// streamed, a share per candidate tile at most.
#[test_case(Metric::SquaredL2, NONE, None; "squared l2")]
#[test_case(Metric::ShiftedL2, NONE, Some(3); "assignment groups")]
#[test_case(Metric::L2, BOTH, Some(2); "l2 floors groups")]
#[test_case(Metric::NegDot, Floors { row: false, col: true }, None; "neg dot col floor")]
#[test_case(Metric::NegDot, Floors { row: true, col: false }, Some(4); "neg dot row floor groups")]
fn shares_merge_to_the_unsplit_answer(metric: Metric, floors: Floors, groups: Option<i32>) {
    let case = Case::draw(70, 230, 24, draw(metric, floors, groups, true, true), 9);
    for base in [NearestCfg::new(32, 16, 2), NearestCfg::streamed(32, 32, 16, 1)] {
        let unsplit = case.interpreted(base);
        case.check("unsplit", &unsplit.0, &unsplit.1, 1e-4, true);
        for splits in [2, 3, 5, case.m.div_ceil(base.bkv)] {
            let cfg = NearestCfg { splits, ..base };
            let split = case.interpreted(cfg);
            case.check(&format!("{cfg:?}"), &split.0, &split.1, 1e-4, true);
            assert_eq!(split, unsplit, "{cfg:?}");
        }
    }
}

/// A share whose candidates are all skipped leaves `+∞` at −1, which the
/// merge passes over: every row's group holds the first `skipped`
/// candidates, so the first shares hold none of a row's; with every one
/// skipped, every row scores `+∞` at −1.
#[test_case(150; "the first shares empty")]
#[test_case(200; "every share empty")]
fn shares_without_a_candidate_leave_infinity(skipped: usize) {
    let mut case = Case::draw(40, 200, 16, draw(Metric::L2, BOTH, Some(2), false, false), 4);
    let (rows, cols) = case.groups.as_mut().unwrap();
    rows.fill(0);
    for (j, g) in cols.iter_mut().enumerate() {
        *g = i32::from(j >= skipped);
    }
    for splits in [2, 4, 7, 13] {
        let cfg = NearestCfg { splits, ..NearestCfg::new(16, 16, 2) };
        let (score, index) = case.interpreted(cfg);
        case.check(&format!("{cfg:?}"), &score, &index, 1e-4, false);
        match skipped == case.m {
            true => assert!(score.iter().all(|s| *s == f64::INFINITY) && index.iter().all(|i| *i == -1.0)),
            false => assert!(index.iter().all(|&i| i >= skipped as f64), "{cfg:?}: {index:?}"),
        }
    }
}

/// The merge folds the shares in order: one with no candidate (`+∞` at −1)
/// is passed over, equal scores keep the earlier share's index, and the
/// least comes out in the metric's units (the L2 family's doubled, L2's
/// square root taken); rows past a block of the merge are ragged.
#[test_case(Metric::NegDot; "neg dot")]
#[test_case(Metric::SquaredL2; "squared l2")]
#[test_case(Metric::L2; "l2")]
fn the_merge_folds_shares_in_order(metric: Metric) {
    let (n, splits, inf) = (300, 3, f64::INFINITY);
    let stride = share_stride(n);
    let (mut score, mut index) = (vec![0.0; splits * stride], vec![0.0; splits * stride]);
    let mut want = vec![];
    for i in 0..n {
        let parts: [(f64, f64); 3] = match i % 4 {
            0 => [(inf, -1.0), (inf, -1.0), (inf, -1.0)],
            1 => [(inf, -1.0), (2.5, 40.0), (2.5, 90.0)],
            2 => [(1.0, 3.0), (-0.5, 41.0), (0.75, 95.0)],
            _ => [(0.25, 7.0), (inf, -1.0), (0.25, 99.0)],
        };
        let parts = parts.map(|(v, j)| if metric == Metric::NegDot { (v, j) } else { (v.abs(), j) });
        for (s, (v, j)) in parts.into_iter().enumerate() {
            (score[s * stride + i], index[s * stride + i]) = (v, j);
        }
        let (v, j) = parts.into_iter().fold((inf, -1.0), |(b, at), (v, j)| if v < b { (v, j) } else { (b, at) });
        let v = match metric {
            Metric::NegDot => v,
            Metric::L2 => round_to(ScalarDType::Float32, (2.0 * v).sqrt()),
            _ => 2.0 * v,
        };
        want.push((v, j));
    }
    let (score, index) = merged(&MergeSpec { n, splits, metric }, score, index);
    for (i, want) in want.into_iter().enumerate() {
        assert_eq!((score[i], index[i]), want, "row {i}");
    }
}

/// The planner splits a streamed tile's candidates at a large self-join,
/// where the blocks in flight span more query blocks than a group, and a
/// resident tile's only where its query blocks leave SMs idle; every share
/// keeps four candidate tiles. The tuner times each admitted config split
/// where it splits, whole where not, and the first whole as well.
#[test_case(crate::atoms::sm86(); "sm_86")]
#[test_case(crate::sweep::amd(AmdArch::Gfx1201, 64); "gfx1201")]
#[test_case(crate::sweep::amd(AmdArch::Gfx1151, 40); "gfx1151")]
fn splits_serve_rereads_and_idle_sms(target: Target) {
    let planner = super::planner(target);
    let (streamed, resident, big) = (NearestCfg::streamed(128, 128, 32, 2), NearestCfg::new(128, 32, 2), 1 << 15);
    let split = |n, m, d, cfg| planner.nearest_split(n, m, d, &cfg);
    let s = split(big, big, 768, streamed).expect("a streamed self-join splits");
    assert!(s.is_power_of_two() && big.div_ceil(streamed.bkv) / s >= 4, "{s}");
    assert_eq!(split(big, big, 128, resident), None, "a resident self-join fills the SMs whole");
    assert!(split(300, big, 128, resident).is_some(), "three query blocks leave SMs idle");
    assert_eq!(split(big, 7 * 128, 768, streamed), None, "seven candidate tiles make one share of four");
    let admitted = [streamed, NearestCfg::streamed(64, 64, 32, 2), resident];
    let timed = planner.nearest_splits(big, big, 768, &admitted);
    let whole: Vec<NearestCfg> = timed.iter().filter(|c| c.splits == 1).copied().collect();
    assert_eq!(whole, [streamed, resident], "{timed:?}");
    assert_eq!(timed.len(), 4, "{timed:?}");
}

// ---- device ------------------------------------------------------------------

/// Every config of the first eight tiles ranked for the case's shape, lowered
/// and run on the device, against the f64 reference as the interpreter is:
/// exactly where the values are integers.
#[test_case(300, 500, 64, draw(Metric::ShiftedL2, NONE, None, false, true); "assignment d64")]
#[test_case(257, 333, 5, draw(Metric::L2, BOTH, Some(3), false, false); "l2 floors groups d5")]
#[test_case(130, 270, 37, draw(Metric::L2, BOTH, Some(2), true, true); "integer ties d37")]
#[test_case(64, 1000, 128, draw(Metric::NegDot, Floors { row: false, col: true }, Some(4), false, false); "neg dot d128")]
#[test_case(100, 200, 32, draw(Metric::SquaredL2, NONE, None, false, false); "squared l2 d32")]
#[test_case(200, 300, 512, draw(Metric::NegDot, NONE, None, false, false); "streamed neg dot d512")]
#[test_case(130, 170, 1000, draw(Metric::L2, BOTH, Some(3), true, true); "streamed integer ties d1000")]
#[test_case(150, 250, 768, draw(Metric::SquaredL2, NONE, Some(2), false, false); "streamed squared l2 groups d768")]
fn nearest_matches_on_device(n: usize, m: usize, d: usize, draw: Draw) {
    let Some(planner) = super::device_planner() else { return };
    let case = Case::draw(n, m, d, draw, 7);
    let depth = planner.nearest_depth(d).unwrap();
    let ranked = planner.nearest_ranked(n, m, depth, draw.metric, draw.floors, draw.groups.is_some());
    let (want_score, want_index) = case.interpreted(ranked[0]);
    case.check("interpreter", &want_score, &want_index, 1e-4, draw.integer);
    for cfg in split_too(&planner, &case, depth, &ranked) {
        let (score, index) = case.on_device(&planner, depth, cfg);
        case.check(&format!("{cfg:?}"), &score, &index, 1e-3, draw.integer);
        let same = index.iter().zip(&want_index).filter(|(a, b)| a == b).count();
        eprintln!("{cfg:?}: {same} of {n} indices as interpreted");
    }
}

/// The first eight tiles of `ranked` as the tuner times them (each unsplit,
/// then at the planner's split where there is one) and the first also split
/// two and three ways.
fn split_too(planner: &Planner, case: &Case, depth: usize, ranked: &[NearestCfg]) -> Vec<NearestCfg> {
    let mut cfgs = planner.nearest_splits(case.n, case.m, depth, &first_eight(ranked));
    cfgs.extend([2, 3].map(|splits| shared_out(NearestCfg { splits, ..ranked[0] }, case.m)));
    cfgs
}

/// A resident trip finds its least's column only in the warps it lowers some
/// row of, by a vote; a streamed tile, folded once per depth pipeline, finds
/// it straight.
#[test_case(crate::sweep::amd(AmdArch::Gfx1201, 64), "@llvm.amdgcn.ballot.i32("; "gfx1201")]
#[test_case(crate::atoms::sm86(), "@llvm.nvvm.vote.any.sync("; "sm_86")]
fn a_resident_trip_votes_before_it_places_its_least(target: Target, vote: &str) {
    let device = crate::sweep::declared(target);
    let case = |cfg| NearestSpec { n: 300, m: 500, d: 64, metric: Metric::NegDot, floors: NONE, groups: false, cfg };
    for (cfg, votes) in [(NearestCfg::new(64, 32, 2), true), (NearestCfg::streamed(64, 32, 32, 2), false)] {
        let ir = crate::sweep::render(nearest::<BF16>(&case(cfg)), &cfg.lowering(&device));
        assert_eq!(ir.contains(vote), votes, "{cfg:?}");
    }
}

/// [`the_trip_that_lowers_a_row_places_its_least`] for every config of the
/// first eight tiles ranked on the device, where warps whose rows no trip
/// lowers skip finding the least's column: exactly the reference's index.
#[test_case(Order::Falling; "the last candidate")]
#[test_case(Order::Rising; "the first candidate")]
#[test_case(Order::Flat; "every candidate alike")]
fn the_trip_that_lowers_a_row_places_its_least_on_device(order: Order) {
    let Some(planner) = super::device_planner() else { return };
    for d in [16, 512] {
        let case = ordered(150, 230, d, order);
        let ranked = planner.nearest_ranked(case.n, case.m, d, Metric::NegDot, NONE, false);
        for cfg in split_too(&planner, &case, d, &ranked) {
            let (score, index) = case.on_device(&planner, d, cfg);
            case.check(&format!("{cfg:?}"), &score, &index, 0.0, true);
        }
    }
}

/// `t` on the host as f64.
fn host(t: &Tensor) -> Vec<f64> {
    t.cast(DType::Float32).to_vec::<f32>().unwrap().into_iter().map(f64::from).collect()
}

/// The op takes the kernel and agrees with its graph on the reference: an
/// assignment at a depth the kernel pads, squared distances, L2 over a
/// self-join under floors (each row may find itself) and under groups too
/// (which skip each row itself), the largest inner product under groups at a
/// deep row, and at the depths of embeddings, whose rows stream.
#[test_case(300, 500, 5, draw(Metric::ShiftedL2, NONE, None, false, false), false, 1; "assignment d5")]
#[test_case(257, 1000, 37, draw(Metric::SquaredL2, NONE, None, false, false), false, 1; "squared l2 d37")]
#[test_case(200, 200, 64, draw(Metric::L2, BOTH, None, false, false), true, 1; "l2 floors self-join")]
#[test_case(300, 300, 16, draw(Metric::L2, BOTH, Some(5), false, false), true, 1; "l2 floors groups self-join")]
#[test_case(100, 130, 128, draw(Metric::NegDot, NONE, Some(3), false, false), false, 1; "neg dot groups d128")]
#[test_case(500, 700, 768, draw(Metric::NegDot, NONE, None, false, false), false, 1; "streamed neg dot d768")]
#[test_case(300, 300, 1000, draw(Metric::SquaredL2, BOTH, Some(4), false, false), true, 1; "streamed self-join d1000")]
#[test_case(300, 300, 16, draw(Metric::L2, BOTH, Some(5), false, false), true, 3; "l2 floors groups self-join split")]
#[test_case(500, 700, 768, draw(Metric::NegDot, NONE, None, false, false), false, 2; "streamed neg dot d768 split")]
fn nearest_matches_the_graph(n: usize, m: usize, d: usize, draw: Draw, self_join: bool, splits: usize) {
    super::ops::untuned(|| {
        let mut case = Case::draw(n, m, d, draw, 11);
        if self_join {
            (case.c, case.m, case.col_floor) = (case.x.clone(), case.n, case.row_floor.clone());
            if let Some((r, c)) = &mut case.groups {
                *c = r.clone();
            }
        }
        let f32s = |v: &[f64]| Tensor::from_slice(v.iter().map(|&a| a as f32).collect::<Vec<_>>());
        let rows = |v: &[f64], r: usize| f32s(v).cast(DType::BFloat16).try_reshape([r as isize, d as isize]).unwrap();
        let x = rows(&case.x, case.n);
        let c = if self_join { x.clone() } else { rows(&case.c, case.m) };
        let row_floor = case.row_floor.as_deref().map(f32s);
        let col_floor = if self_join { row_floor.clone() } else { case.col_floor.as_deref().map(f32s) };
        let group = |g: &[i32]| Tensor::from_slice(g);
        let groups = case.groups.as_ref().map(|(r, c)| (group(r), group(c)));
        let opts = Nearest {
            metric: draw.metric,
            row_floor: row_floor.as_ref(),
            col_floor: col_floor.as_ref(),
            groups: groups.as_ref().map(|(r, c)| (r, c)),
        };
        // Split, the ranking's first config, its candidates in `splits` shares.
        let split = |_: &Planner, ranked: &[NearestCfg], spec: NearestSpec| {
            let cfg = shared_out(NearestCfg { splits, ..ranked[0] }, spec.m);
            NearestSpec { cfg, ..spec }
        };
        let (score, index) = match splits {
            1 => tk::nearest(&x, &c, opts).unwrap(),
            _ => tk::nearest::picked(&x, &c, opts, split).unwrap(),
        };
        super::ops::assert_kernel(&score, if splits == 1 { "nearest" } else { "nearest_merge" });
        case.check("kernel", &host(&score), &host(&index), 1e-3, false);
        let (score, index) = tk::nearest::graph(&x, &c, opts).unwrap();
        case.check("graph", &host(&score), &host(&index), 1e-3, false);
    });
}

/// Pairs a second over a self-join of `ROWS` rows, of the tuner's pick and
/// every config it admitted, per depth: squared distances, the same under
/// groups (64 of them), L2 above both floors; beside the graph at
/// `GRAPH_ROWS` (its `[n, n]` scores do not fit at the kernel's shape).
/// Prints, never asserts (run with `--ignored --nocapture --test-threads=1`).
#[test]
#[ignore = "perf probe: needs a GPU"]
fn nearest_throughput_probe() {
    const ROWS: usize = 1 << 16;
    const GRAPH_ROWS: usize = 1 << 13;
    let Some(planner) = super::device_planner() else { return };
    crate::tune::scoped(true, || {
        for d in [16, 32, 64, 128] {
            for (metric, floors, groups) in
                [(Metric::SquaredL2, NONE, false), (Metric::SquaredL2, NONE, true), (Metric::L2, BOTH, false)]
            {
                time_nearest(&planner, ROWS, d, metric, floors, groups);
                match graph_seconds(GRAPH_ROWS, d, metric, floors, groups) {
                    Some(s) => eprintln!(
                        "  graph at {GRAPH_ROWS}: {:8.2} ms {:7.1} Gpairs/s",
                        s * 1e3,
                        (GRAPH_ROWS * GRAPH_ROWS) as f64 / s / 1e9
                    ),
                    None => eprintln!("  graph at {GRAPH_ROWS}: failed"),
                }
            }
        }
    });
}

/// Pairs a second and TFLOP/s (`2·d` a pair) over a self-join of `ROWS`
/// rows at the depths of embeddings, whose rows stream, of the tuner's pick
/// and every config it admitted: squared distances and the inner product
/// (cosine over normalized rows); beside the tuner's GEMM pick on the same
/// device at `8192 × 8192 × d` and at 4096³, timed alike. `NEAREST_ROWS`
/// sets other rows and `NEAREST_DEPTHS` (comma-separated) other depths.
/// Prints, never asserts (run with `--ignored --nocapture --test-threads=1`).
#[test]
#[ignore = "perf probe: needs a GPU"]
fn nearest_wide_probe() {
    let var = |name| std::env::var(name).ok();
    let rows = var("NEAREST_ROWS").map_or(1 << 16, |v| v.parse().expect("NEAREST_ROWS is a row count"));
    let depths: Vec<usize> = var("NEAREST_DEPTHS").map_or(vec![256, 512, 768, 1024], |v| {
        v.split(',').map(|d| d.trim().parse().expect("NEAREST_DEPTHS lists depths")).collect()
    });
    let Some(planner) = super::device_planner() else { return };
    crate::tune::scoped(true, || {
        for d in depths {
            for metric in [Metric::SquaredL2, Metric::NegDot] {
                time_nearest(&planner, rows, d, metric, NONE, false);
            }
            time_gemm(&planner, 8192, 8192, d);
        }
        time_gemm(&planner, 4096, 4096, 4096);
    });
}

/// Times the tuner's pick for an `n`-row self-join and every config it
/// admitted, printing each one's pairs a second and TFLOP/s.
fn time_nearest(planner: &Planner, n: usize, d: usize, metric: Metric, floors: Floors, groups: bool) {
    let (admitted, pick) = device_nearest(planner, n, d, metric, floors, groups);
    let mut timed = admitted.clone();
    if !timed.contains(&pick) {
        timed.push(pick);
    }
    let spec = |cfg| NearestSpec { n, m: n, d, metric, floors, groups, cfg };
    let ns = super::measure(timed.iter().map(|&cfg| tk::nearest::programs(planner, DType::BFloat16, &spec(cfg))));
    eprintln!("== nearest {n}x{n}x{d} {metric:?} {floors:?} groups={groups}: {} admitted", admitted.len());
    let pairs = (n * n) as f64;
    for (cfg, ns) in timed.iter().zip(ns) {
        let mark = if *cfg == pick { " (tuner's pick)" } else { "" };
        let rate = ns.map_or("failed".into(), |ns| {
            let ns = ns as f64;
            format!("{:8.2} ms {:7.1} Gpairs/s {:6.2} TFLOP/s", ns / 1e6, pairs / ns, 2.0 * d as f64 * pairs / ns / 1e3)
        });
        eprintln!("  {cfg:?}: {rate}{mark}");
    }
}

/// Times the tuner's GEMM pick for `m × n × k`, as the tuner times it.
fn time_gemm(planner: &Planner, m: usize, n: usize, k: usize) {
    use crate::kernels::Batch;
    use crate::kernels::gemm::{Epilogue, GemmSpec, gemm};
    let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg: super::device::BIG };
    let (_, cfg) = super::device::device_gemms(planner, &spec);
    let [ns] = super::measure([vec![(gemm::<BF16>(&GemmSpec { cfg, ..spec }), cfg.lowering(&planner.device))]])[..]
    else {
        unreachable!("one candidate")
    };
    let rate = ns.map_or("failed".into(), |ns| {
        let ns = ns as f64;
        format!("{:8.2} ms {:6.2} TFLOP/s", ns / 1e6, 2.0 * (m * n * k) as f64 / ns / 1e3)
    });
    eprintln!("== gemm {m}x{n}x{k} {cfg:?}: {rate} (tuner's pick)");
}

/// The configs the tuner admits for an `n`-row self-join on the device, and its pick.
fn device_nearest(
    planner: &Planner,
    n: usize,
    d: usize,
    metric: Metric,
    floors: Floors,
    groups: bool,
) -> (Vec<NearestCfg>, NearestCfg) {
    let ranked = planner.nearest_ranked(n, n, d, metric, floors, groups);
    let spec = |cfg| NearestSpec { n, m: n, d, metric, floors, groups, cfg };
    let build = |cfg: NearestCfg| tk::nearest::programs(planner, DType::BFloat16, &spec(cfg));
    let shape = [n, n, d];
    let (admitted, pick) = super::tuned_on_device(
        planner,
        ("nearest", DType::BFloat16, &shape, (crate::ops::DEVICE_TEST, metric, floors, groups)),
        &ranked,
        |admitted| planner.nearest_splits(n, n, d, admitted),
        build,
    );
    (admitted, pick.expect("a nearest kernel the planner runs"))
}

/// The best of a few runs of the graph fallback's indices for an `n`-row
/// self-join at depth `d`, in seconds.
fn graph_seconds(n: usize, d: usize, metric: Metric, floors: Floors, groups: bool) -> Option<f64> {
    let mut seed = 5;
    let data: Vec<f32> = (0..n * d).map(|_| super::lcg(&mut seed) as f32).collect();
    let x = Tensor::from_slice(&data).cast(DType::BFloat16).try_reshape([n as isize, d as isize]).ok()?.contiguous();
    let floor = Tensor::from_slice(vec![0.5f32; n]);
    let group = Tensor::from_slice((0..n as i32).map(|i| i % 64).collect::<Vec<_>>());
    x.realize().ok()?;
    let opts = Nearest {
        metric,
        row_floor: floors.row.then_some(&floor),
        col_floor: floors.col.then_some(&floor),
        groups: groups.then_some((&group, &group)),
    };
    let (_, index) = tk::nearest::graph(&x, &x, opts).ok()?;
    let plan = index.prepare().ok()?;
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        best = best.min(crate::tune::run(&plan)?.as_secs_f64());
    }
    Some(best)
}

/// How a row's minimum folds under each way of issuing the product: on a
/// square core (RDNA, CDNA) the swapped issue holds a row's candidates along
/// a lane's registers, one or two lane moves a reduction where the direct
/// issue spreads a row over 16 lanes (four, for each row a lane holds);
/// CUDA's `m16n8k16` issues one way, a row over the four lanes of a quad. No
/// relayout or scattered load tells the two apart, so the folds' lane moves
/// decide: the inference swaps where the core can.
#[test_case(crate::atoms::sm86(), 2, None; "sm_86")]
#[test_case(crate::sweep::amd(AmdArch::Gfx1201, 64), 4, Some(1); "gfx1201")]
#[test_case(crate::sweep::amd(AmdArch::Gfx1100, 96), 4, Some(1); "gfx1100")]
#[test_case(crate::sweep::amd(AmdArch::Gfx1151, 40), 4, Some(1); "gfx1151")]
#[test_case(crate::sweep::amd(AmdArch::Gfx942, 304), 4, Some(2); "gfx942")]
fn a_swapped_product_folds_a_row_in_lanes(target: Target, direct: usize, swapped: Option<usize>) {
    use crate::ir::{Block, Stmt, TileOp};
    let how = target.lane_move;
    let planner = super::planner(target.clone());
    let cfg = NearestCfg { warp_rows: 32, ..NearestCfg::new(128, 64, 2) };
    let spec = NearestSpec { n: 1000, m: 1000, d: 64, metric: Metric::SquaredL2, floors: NONE, groups: false, cfg };
    let grid = crate::layouts::WarpGrid { rows: cfg.warps(), cols: 1 };
    fn pin(block: &mut Block, to: Orient) {
        for s in &mut block.0 {
            match s {
                Stmt::Let { op: TileOp::Mma { orient, .. }, .. } => *orient = Some(to),
                other => other.blocks_mut().for_each(|b| pin(b, to)),
            }
        }
    }
    // Each reduction's fold masks and the lane moves a lane issues for it,
    // and the product's orientation.
    let folds = |orient: Option<Orient>| {
        let mut prog = nearest::<BF16>(&spec);
        if let Some(o) = orient {
            pin(&mut prog.body, o);
        }
        let laid = crate::layouts::infer(prog, &planner.device, grid, Core::Warp).unwrap();
        let (mut masks, mut moves, mut issued) = (vec![], vec![], vec![]);
        for (_, s) in laid.prog.walk() {
            match s {
                Stmt::Let { op: TileOp::Mma { orient, .. }, .. } => issued.push(orient.unwrap()),
                Stmt::Let { dst, op: TileOp::Reduce { src, axis, .. } } => {
                    let (ls, ld) =
                        (laid.layouts[src.index()].as_ref().unwrap(), laid.layouts[dst.index()].as_ref().unwrap());
                    masks.push(ls.fold_masks(*axis).len());
                    moves.push(ls.fold_moves(ld, *axis, how));
                }
                Stmt::Let { op: TileOp::Relayout { .. }, .. } => panic!("a relayout"),
                _ => {}
            }
        }
        (masks, moves, issued)
    };
    let as_direct = folds(Some(Orient::Direct));
    assert_eq!(as_direct.0, vec![direct; 2], "direct");
    let Some(swapped) = swapped else {
        assert_eq!(folds(None), as_direct, "as inferred: one way to issue");
        return;
    };
    let as_swapped = folds(Some(Orient::Swapped));
    assert_eq!(as_swapped.0, vec![swapped; 2], "swapped");
    assert!(as_swapped.1.iter().zip(&as_direct.1).all(|(s, d)| s < d), "{:?} vs {:?}", as_swapped.1, as_direct.1);
    assert_eq!(folds(None), as_swapped, "as inferred");
}

/// Tuning wall time, the pick's throughput and predicted against measured
/// ns per timed config: nearest NegDot at `TUNE_ROWS` (16384) squared by
/// `TUNE_DEPTHS` (256,512,768,1024), the GEMMs `TUNE_GEMMS` (`MxNxK,…`,
/// 4096³) and the attentions `TUNE_ATTN` (`batch:heads:t:d:causal,…`, none).
/// Run on a store holding only the device's probes (its `value|probe:`
/// lines), so every shape tunes cold; the predicted spread is what
/// [`PRUNED_PAST`](crate::ops::config::PRUNED_PAST) is chosen from. Prints,
/// never asserts (`--ignored --nocapture --test-threads=1`).
#[test]
#[ignore = "perf probe: needs a GPU"]
fn tuning_probe() {
    use crate::kernels::Batch;
    use crate::kernels::attention::{AttnMask, AttnSpec, FaCfg, attention};
    use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
    use crate::ops::config::{Attention, Nearest as Problem, Problem as Gemm};
    let var = |name| std::env::var(name).ok();
    let list = |name, default: &str| -> Vec<Vec<usize>> {
        let text = var(name).unwrap_or(default.into());
        text.split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.split(['x', ':']).map(|v| v.parse().unwrap()).collect())
            .collect()
    };
    let rows: usize = var("TUNE_ROWS").map_or(16384, |v| v.parse().unwrap());
    let depths: Vec<usize> = list("TUNE_DEPTHS", "256,512,768,1024").concat();
    let Some(planner) = super::device_planner() else { return };
    fn report<C: crate::ops::config::Tiled + std::fmt::Debug>(
        planner: &Planner,
        name: String,
        (op, shape, salt): (&'static str, &[usize], impl std::fmt::Debug),
        (ranked, expand): (&[C], impl Fn(&[C]) -> Vec<C>),
        build: impl Fn(C) -> crate::tune::Candidate,
        predicted: impl Fn(&C) -> f64,
        work: f64,
    ) {
        let start = std::time::Instant::now();
        let store = crate::tune::TuneStore::global();
        let mut tuner = crate::ops::Tuner::new(store, &planner.device, DType::BFloat16, shape, salt, &build);
        let admitted = expand(&tuner.admitted(op, &crate::ops::config::representatives(ranked)));
        let ordered = tuner.ordered(op, &admitted);
        let pick = tuner.pipelined(op, &ordered, ranked);
        let wall = start.elapsed().as_secs_f64();
        let [ns] = super::measure([build(pick)])[..] else { unreachable!() };
        let rate =
            ns.map_or("failed".into(), |ns| format!("{:.3} ms {:.2} TFLOP/s", ns as f64 / 1e6, work / ns as f64 / 1e3));
        eprintln!("== {name}: tuning {wall:.2} s, {} admitted, pick {rate} {pick:?}", admitted.len());
        let best_pred = admitted.iter().map(&predicted).fold(f64::INFINITY, f64::min);
        let best_meas = ordered.iter().filter_map(|o| o.1).min().unwrap_or(1) as f64;
        for c in &admitted {
            let meas = ordered.iter().find(|(o, _)| o == c).and_then(|o| o.1);
            let m = meas.map_or("-".into(), |m| format!("{:.3}", m as f64 / best_meas));
            let p = predicted(c);
            eprintln!(
                "  pred {:7.3} meas {m:>7} {:10.3} ms pred {:10.3} ms {c:?}",
                p / best_pred,
                meas.unwrap_or(0) as f64 / 1e6,
                p / 1e6
            );
        }
    }
    crate::tune::scoped(true, || {
        for d in depths {
            let (n, metric, floors) = (rows, Metric::NegDot, NONE);
            let ranked = planner.nearest_ranked(n, n, d, metric, floors, false);
            let spec = |cfg| NearestSpec { n, m: n, d, metric, floors, groups: false, cfg };
            let build = |cfg: NearestCfg| tk::nearest::programs(&planner, DType::BFloat16, &spec(cfg));
            let p = Problem { n, m: n, d, metric, floors, groups: false };
            let salt = (crate::ops::DEVICE_TEST, metric, floors, false);
            let name = format!("nearest {n}x{n}x{d} {metric:?}");
            let work = 2.0 * (n * n * d) as f64;
            report(
                &planner,
                name,
                ("nearest", &[n, n, d], salt),
                (&ranked, |a: &[NearestCfg]| planner.nearest_splits(n, n, d, a)),
                build,
                |c| p.admit(&planner, c).unwrap().0,
                work,
            );
        }
        for s in list("TUNE_GEMMS", "4096x4096x4096") {
            let [m, n, k] = s[..] else { panic!("MxNxK") };
            let ranked = planner.gemm_ranked(1, m, n, k, false);
            let spec =
                GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg: super::device::BIG };
            let build =
                |cfg: GemmCfg| vec![(gemm::<BF16>(&GemmSpec { cfg, ..spec.clone() }), cfg.lowering(&planner.device))];
            let p = Gemm { batches: 1, m, n, k, halves: 1 };
            let work = 2.0 * (m * n * k) as f64;
            let key = ("gemm", &[1, m, n, k][..], crate::ops::DEVICE_TEST);
            report(
                &planner,
                format!("gemm {m}x{n}x{k}"),
                key,
                (&ranked, <[_]>::to_vec),
                build,
                |c| p.admit(&planner, c).unwrap().0,
                work,
            );
        }
        for s in list("TUNE_ATTN", "") {
            let [batch, heads, t, d, causal] = s[..] else { panic!("batch:heads:t:d:causal") };
            let mask = AttnMask { causal: causal == 1, ..AttnMask::default() };
            let spec = super::attention::probe_spec(batch, heads, t, t, d, mask);
            let ranked = planner.attention_ranked(batch * heads, t, t, d, mask.causal, false);
            let build = |cfg: FaCfg| {
                vec![(attention::<BF16>(&AttnSpec { cfg, ..spec.clone() }), cfg.lowering(&planner.device))]
            };
            let p = Attention { tiles: batch * heads, t, tk: t, d, causal: mask.causal, bias: false };
            let work = 4.0 * (batch * heads * t * t * d) as f64 / if mask.causal { 2.0 } else { 1.0 };
            let shape = [batch, t, t, heads, d];
            report(
                &planner,
                format!("attention {s:?}"),
                ("attention", &shape, mask),
                (&ranked, <[_]>::to_vec),
                build,
                |c| p.admit(&planner, c).unwrap().0,
                work,
            );
        }
    });
}
