//! The nearest-candidate kernel: the interpreter against an f64 reference on
//! the host (every metric, floors, groups, ties, rows left with no
//! candidate, lists of a row's least scores), every ranked config against
//! the reference on the device, the op against its graph, and the probes.

use proptest::prelude::*;
use svod_dtype::{AmdArch, DType, ScalarDType};
use svod_tensor::Tensor;
use test_case::test_case;

use super::first_eight;
use crate::atoms::{Core, Target};
use crate::build::{BF16, F16};
use crate::interp::{round_to, run};
use crate::ir::Orient;
use crate::kernels::nearest::{
    Floors, LaneQueue, MergeSpec, Metric, NearestCfg, NearestSpec, SelectCfg, SelectSpec, nearest, nearest_merge,
    nearest_select, share_stride,
};
use crate::ops::config::{self, Planner};
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
    /// `x` and `c` exact in f16 and run as f16, else bf16.
    f16: bool,
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
        Case { n, m, d, metric: draw.metric, x, c, row_floor, col_floor, groups, f16: false }
    }

    /// The case with `x` and `c` non-negative where they are integers, so a
    /// matrix core sums them exactly: gfx1151's sums products of mixed signs
    /// `2^-20` off (a 16-deep integer inner product of −12 comes out
    /// `−12 + 2^-20`), which turns an exact tie into an order.
    fn unsigned(mut self, integer: bool) -> Self {
        if integer {
            self.x.iter_mut().chain(&mut self.c).for_each(|v| *v = v.abs());
        }
        self
    }

    /// The case with `x` and `c` rounded to f16 (exact already but for
    /// bf16's values past f16's precision at the bottom of its range), run
    /// as f16.
    fn in_f16(mut self) -> Self {
        for v in self.x.iter_mut().chain(&mut self.c) {
            *v = round_to(ScalarDType::Float16, *v);
        }
        Self { f16: true, ..self }
    }

    fn floors(&self) -> Floors {
        Floors { row: self.row_floor.is_some(), col: self.col_floor.is_some() }
    }

    /// The planner's problem of the case at `depth`, keeping `k` scores a row.
    fn problem(&self, depth: usize, k: usize) -> config::Nearest {
        let (n, m, metric, floors, groups) = (self.n, self.m, self.metric, self.floors(), self.groups.is_some());
        config::Nearest { n, m, d: depth, metric, floors, groups, k }
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

    /// Each row's `k` least scores ascending and their indices, an equal
    /// score at the smaller index; `(+∞, −1)` past the row's admissible
    /// candidates.
    fn reference_list(&self, k: usize) -> Vec<Vec<(f64, i64)>> {
        (0..self.n)
            .map(|i| {
                let mut row: Vec<(f64, i64)> =
                    (0..self.m).map(|j| (self.pair(i, j), j as i64)).filter(|p| p.0 < f64::INFINITY).collect();
                // Not `total_cmp`: −0 ties +0, as the kernel compares them.
                row.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("no NaN").then(a.1.cmp(&b.1)));
                row.resize(k.max(row.len()), (f64::INFINITY, -1));
                row.truncate(k);
                row
            })
            .collect()
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

    /// Per candidate, the first one holding the same row, floor and group.
    fn alikes(&self) -> Vec<usize> {
        // Equal values alike, −0 as +0.
        let bits = |v: f64| (v + 0.0).to_bits();
        let mut first: std::collections::HashMap<Vec<u64>, usize> = std::collections::HashMap::new();
        (0..self.m)
            .map(|j| {
                let mut key: Vec<u64> = Self::row(&self.c, self.d, j).iter().map(|&v| bits(v)).collect();
                key.extend(self.col_floor.as_ref().map(|f| bits(f[j])));
                key.extend(self.groups.as_ref().map(|(_, g)| g[j] as u64));
                *first.entry(key).or_insert(j)
            })
            .collect()
    }

    /// The kernel's parameters at `depth` (zero columns past `d`), the norms
    /// in f32 as the op computes them, then the two outputs of `k` scores a
    /// row (of every share, split `splits` ways).
    fn params(&self, depth: usize, splits: usize, k: usize) -> Vec<Vec<f64>> {
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
        let parts = k * if splits == 1 { self.n } else { splits * share_stride(self.n) };
        params.extend([vec![0.0; parts], vec![0.0; parts]]);
        params
    }

    fn spec(&self, depth: usize, cfg: NearestCfg, k: usize) -> NearestSpec {
        let (n, m, metric, floors, groups) = (self.n, self.m, self.metric, self.floors(), self.groups.is_some());
        NearestSpec { n, m, d: depth, metric, floors, groups, k, cfg, live: Default::default() }
    }

    /// The kernel of `k` scores a row under `cfg` at `depth`, in the case's type.
    fn program(&self, depth: usize, cfg: NearestCfg, k: usize) -> crate::ir::Program {
        let spec = self.spec(depth, cfg, k);
        if self.f16 { nearest::<F16>(&spec) } else { nearest::<BF16>(&spec) }
    }

    fn merge(&self, splits: usize, k: usize) -> MergeSpec {
        MergeSpec { n: self.n, splits, metric: self.metric, k, live: Default::default() }
    }

    /// The scores and indices the interpreter computes under `cfg`, split,
    /// its shares merged.
    fn interpreted(&self, cfg: NearestCfg) -> (Vec<f64>, Vec<f64>) {
        self.listed(cfg, 1)
    }

    /// [`Self::interpreted`] keeping `k` scores a row, `[n, k]`; a score
    /// matrix's in the blocks the op takes.
    fn listed(&self, cfg: NearestCfg, k: usize) -> (Vec<f64>, Vec<f64>) {
        if cfg.select.is_some() {
            let blocks = crate::kernels::nearest::select_blocks(self.n, self.m, k, &cfg);
            let device = crate::sweep::declared(crate::atoms::sm86());
            return self.selected(cfg, k, blocks, &device, &|prog, _, params| run(&prog, params, &[]).unwrap());
        }
        let depth = self.d.next_multiple_of(16);
        let mut out = run(&self.program(depth, cfg, k), self.params(depth, cfg.splits, k), &[]).unwrap();
        let (index, score) = (out.pop().unwrap(), out.pop().unwrap());
        if cfg.splits == 1 {
            return (score, index);
        }
        merged(&self.merge(cfg.splits, k), score, index)
    }

    /// [`Self::listed`] through a score matrix (`cfg.select`), its rows
    /// listed by [`nearest_select`] in panels of `panel` rows and blocks of
    /// `cols` candidates, the lists carried from block to block; each
    /// program `run` on its lowering.
    fn selected(
        &self,
        cfg: NearestCfg,
        k: usize,
        (panel, cols): (usize, usize),
        device: &crate::atoms::Device,
        run: &impl Fn(crate::ir::Program, &crate::lower::Lowering, Vec<Vec<f64>>) -> Vec<Vec<f64>>,
    ) -> (Vec<f64>, Vec<f64>) {
        let depth = self.d.next_multiple_of(16);
        let select = cfg.select.expect("a score matrix");
        let (scored, listing) = (cfg.lowering(device), select.lowering(device));
        let (score, index) = (0..self.n).step_by(panel).fold((vec![], vec![]), |(mut score, mut index), r0| {
            let rows = panel.min(self.n - r0);
            let mut lists = vec![vec![0.0; rows * k], vec![0.0; rows * k]];
            for c0 in (0..self.m).step_by(cols) {
                let width = cols.min(self.m - c0);
                let part = self.part(r0..r0 + rows, c0..c0 + width);
                let mut params = part.params(depth, 1, k);
                params.truncate(params.len() - 2);
                params.push(vec![0.0; rows * width]);
                let matrix = run(part.program(depth, cfg, k), &scored, params).pop().unwrap();
                let spec = SelectSpec {
                    n: rows,
                    cols: width,
                    k,
                    metric: self.metric,
                    first: c0 == 0,
                    cfg: select,
                    live: Default::default(),
                };
                let chunk = vec![c0 as f64, f64::from(c0 + width == self.m)];
                let carried = if c0 == 0 { vec![] } else { lists };
                let params = [vec![matrix, chunk], carried, vec![vec![0.0; rows * k]; 2]].concat();
                let mut out = run(nearest_select(&spec), &listing, params);
                lists = out.split_off(out.len() - 2);
            }
            score.extend(&lists[0]);
            index.extend(&lists[1]);
            (score, index)
        });
        (score, index)
    }

    /// The case of rows `rows` against candidates `cols`.
    fn part(&self, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> Case {
        let d = self.d;
        let (n, m) = (rows.len(), cols.len());
        let slice = |v: &[f64], r: &std::ops::Range<usize>, w: usize| v[r.start * w..r.end * w].to_vec();
        Case {
            n,
            m,
            x: slice(&self.x, &rows, d),
            c: slice(&self.c, &cols, d),
            row_floor: self.row_floor.as_ref().map(|f| slice(f, &rows, 1)),
            col_floor: self.col_floor.as_ref().map(|f| slice(f, &cols, 1)),
            groups: self.groups.as_ref().map(|(r, c)| (r[rows.clone()].to_vec(), c[cols.clone()].to_vec())),
            ..self.clone()
        }
    }

    /// The scores and indices of `k` a row `cfg` computes on the device
    /// (split, its shares merged there).
    fn on_device(&self, planner: &Planner, depth: usize, cfg: NearestCfg, k: usize) -> (Vec<f64>, Vec<f64>) {
        if cfg.select.is_some() {
            // Two panels of rows and three blocks of candidates: lists carried.
            let blocks = (self.n.div_ceil(2), self.m.div_ceil(3));
            return self.selected(cfg, k, blocks, &planner.device, &|prog, lowering, params| {
                let outs = super::targets::on_device(prog, lowering, &params);
                outs.into_iter().zip(params).map(|(o, p)| o.unwrap_or(p)).collect()
            });
        }
        let (prog, params) = (self.program(depth, cfg, k), self.params(depth, cfg.splits, k));
        let mut outs = super::targets::on_device(prog, &cfg.lowering(&planner.device), &params);
        let (index, score) = (outs.pop().flatten().unwrap(), outs.pop().flatten().unwrap());
        if cfg.splits == 1 {
            return (score, index);
        }
        let merge = self.merge(cfg.splits, k);
        let params = [score, index, vec![0.0; self.n * k], vec![0.0; self.n * k]];
        let mut outs = super::targets::on_device(nearest_merge(&merge), &merge.lowering(&planner.device), &params);
        let index = outs.pop().flatten().unwrap();
        (outs.pop().flatten().unwrap(), index)
    }

    /// `score` and `index` (`[n, k]`) are each row's `k` least and where:
    /// every slot's score within `tol` of the reference's (as [`Self::check`]
    /// compares them), `+∞` at −1 exactly where the reference's is; the
    /// scores ascending; every index admissible, distinct in its row,
    /// scoring the slot's reference score within `tol`, and preceded in its
    /// row by every earlier candidate alike (an equal score at the smaller
    /// index); and, where `exact`, the reference's own.
    fn check_list(&self, what: &str, k: usize, score: &[f64], index: &[f64], tol: f64, exact: bool) {
        let want = self.reference_list(k);
        let alike = self.alikes();
        assert_eq!((score.len(), index.len()), (self.n * k, self.n * k), "{what}: rows");
        let l2 = self.metric == Metric::L2;
        let near = |a: f64, b: f64| {
            let (a, b) = if l2 { (a * a, b * b) } else { (a, b) };
            a == b || (a - b).abs() <= tol * b.abs().max(1.0)
        };
        for (i, want) in want.iter().enumerate() {
            let (s, at) = (&score[i * k..][..k], &index[i * k..][..k]);
            assert!(s.is_sorted(), "{what}: row {i} scores {s:?}");
            for (slot, (&(ws, wj), (&s, &j))) in want.iter().zip(s.iter().zip(at)).enumerate() {
                let (row, wants) = (format!("{what}: row {i} slot {slot}"), format!("want {ws} at {wj}"));
                assert!(near(s, ws), "{row} scores {s}, {wants}");
                if ws.is_infinite() {
                    assert_eq!(j, -1.0, "{row} has no candidate");
                    continue;
                }
                assert!(j >= 0.0 && (j as usize) < self.m, "{row} picks {j}");
                let j = j as usize;
                assert!(near(self.pair(i, j), ws), "{row} picks {j} scoring {}, {wants}", self.pair(i, j));
                assert!(!at[..slot].contains(&(j as f64)), "{row} picks {j} twice: {at:?}");
                let earlier = (alike[j]..j).filter(|&a| alike[a] == alike[j]);
                for a in earlier {
                    assert!(at[..slot].contains(&(a as f64)), "{row} picks {j} before its alike {a}: {at:?}");
                }
                if exact {
                    assert_eq!(j as i64, wj, "{row}: {at:?}");
                }
            }
        }
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
        let alike = self.alikes();
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
            assert_eq!(alike[j], j, "{what}: row {i} picks {j}, a repeat of an earlier candidate");
            if exact {
                assert_eq!(j as i64, at[i], "{what}: row {i}");
            }
        }
    }
}

/// The interpreter's merge of a split kernel's shares.
fn merged(spec: &MergeSpec, score: Vec<f64>, index: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let rows = spec.n * spec.k;
    let params = vec![score, index, vec![0.0; rows], vec![0.0; rows]];
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
    let (score, index) = merged(&MergeSpec { n, splits, metric, k: 1, live: Default::default() }, score, index);
    for (i, want) in want.into_iter().enumerate() {
        assert_eq!((score[i], index[i]), want, "row {i}");
    }
}

/// The scores a row's list keeps in the tests: its least alone, short
/// lists, and the longest a register list holds.
const KS: [usize; 6] = [1, 2, 4, 8, 16, config::REGISTER_K];

proptest! {
    /// The program keeps each row's `k` least scores, ascending, and their
    /// indices over any shape (ragged rows and candidates, padded and
    /// overhanging depths, shares merged), every metric, either floor, with
    /// and without groups (lists longer than a row's admissible candidates
    /// included), exact ties, in bf16 and f16.
    #[test]
    fn lists_match_the_reference(
        n in 1usize..50,
        m in 1usize..90,
        d in 1usize..40,
        draw in draws(),
        cfg in small_cfg(),
        k in prop::sample::select(KS.to_vec()),
        f16 in any::<bool>(),
        seed in any::<u64>(),
    ) {
        let case = Case::draw(n, m, d, draw, seed);
        let case = if f16 { case.in_f16() } else { case };
        let (score, index) = case.listed(shared_out(cfg, m), k);
        case.check_list("interpreter", k, &score, &index, 1e-4, draw.integer);
    }
}

/// Every regime keeps the reference's lists, the shares' bit for bit the
/// unsplit ones: resident and streamed, split two and five ways and a share
/// per candidate tile, for every `k` of [`KS`] in bf16 and f16, under every
/// metric, floors and groups, with exact ties across every trip and share
/// boundary (each candidate repeats one of the first three, so a list of
/// more than three holds repeats, in the order of their indices).
#[test_case(Metric::SquaredL2, NONE, None; "squared l2")]
#[test_case(Metric::ShiftedL2, NONE, Some(3); "assignment groups")]
#[test_case(Metric::L2, BOTH, Some(2); "l2 floors groups")]
#[test_case(Metric::NegDot, Floors { row: false, col: true }, None; "neg dot col floor")]
#[test_case(Metric::NegDot, Floors { row: true, col: false }, Some(4); "neg dot row floor groups")]
fn lists_match_in_every_regime(metric: Metric, floors: Floors, groups: Option<i32>) {
    let bf16 = Case::draw(50, 150, 24, draw(metric, floors, groups, true, true), 9);
    for case in [bf16.clone(), bf16.in_f16()] {
        for k in KS {
            for base in [NearestCfg::new(32, 16, 2), NearestCfg::streamed(32, 32, 16, 1)] {
                let unsplit = case.listed(base, k);
                case.check_list(&format!("{base:?} k{k}"), k, &unsplit.0, &unsplit.1, 1e-4, true);
                for splits in [2, 5, case.m.div_ceil(base.bkv)] {
                    let cfg = NearestCfg { splits, ..base };
                    let split = case.listed(cfg, k);
                    case.check_list(&format!("{cfg:?} k{k}"), k, &split.0, &split.1, 1e-4, true);
                    assert_eq!(split, unsplit, "{cfg:?} k{k}");
                }
            }
        }
    }
}

/// Lists longer than a register list holds: just past it, powers of two
/// up to a list of 1024, and one between.
const LONG_KS: [usize; 6] = [33, 64, 100, 128, 256, 1024];

/// A long list's ways: queued 16 or 32 slots a row into the outputs, or
/// through a score matrix listed by one to four warps a block, a lane
/// queuing one to eight scores (as many as the list's runs allow).
fn long_cfgs(k: usize) -> Vec<NearestCfg> {
    let queued = [(16, NearestCfg::new(32, 16, 2)), (32, NearestCfg::streamed(32, 32, 16, 1))];
    let queued = queued.map(|(q, cfg)| NearestCfg { queue: Some(q), ..cfg });
    let runs = crate::kernels::nearest::select_runs(k);
    let selects = [(1, 1), (2, 2), (4, 8)].map(|(warps, queue)| SelectCfg { warps, queue: queue.min(runs) });
    let bases = [NearestCfg::new(16, 16, 2), NearestCfg::streamed(32, 32, 16, 2), NearestCfg::new(64, 64, 1)];
    let matrices = selects.into_iter().zip(bases).map(|(s, cfg)| NearestCfg { select: Some(s), ..cfg });
    queued.into_iter().chain(matrices).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    /// A long list, queued or through a score matrix (in panels of any rows
    /// and blocks of any candidates, the lists carried from block to block),
    /// keeps each row's `k` least ascending and their indices over any
    /// shape, every metric, either floor, with and without groups, exact
    /// ties, in bf16 and f16; past a row's admissible candidates `+∞` at −1.
    #[test]
    fn long_lists_match_the_reference(
        n in 1usize..40,
        m in 1usize..300,
        d in 1usize..40,
        draw in draws(),
        k in prop::sample::select(LONG_KS.to_vec()),
        way in 0usize..5,
        panel in 1usize..40,
        cols in 1usize..300,
        f16 in any::<bool>(),
        seed in any::<u64>(),
    ) {
        let case = Case::draw(n, m, d, draw, seed);
        let case = if f16 { case.in_f16() } else { case };
        let cfg = long_cfgs(k)[way];
        let (score, index) = match cfg.select {
            None => case.listed(cfg, k),
            Some(_) => {
                let device = crate::sweep::declared(crate::atoms::sm86());
                case.selected(cfg, k, (panel, cols), &device, &|prog, _, params| run(&prog, params, &[]).unwrap())
            }
        };
        case.check_list("interpreter", k, &score, &index, 1e-4, draw.integer);
    }
}

/// Every way of a long list keeps the reference's, each the others' bit for
/// bit: queued resident and streamed, through a score matrix whole and in
/// ragged panels and blocks, for every `k` of [`LONG_KS`] (most past the
/// candidates) in bf16 and f16, under every metric, floors and groups, with
/// exact ties across every trip, queue and block (each candidate repeats one
/// of the first three).
#[test_case(Metric::SquaredL2, NONE, None; "squared l2")]
#[test_case(Metric::ShiftedL2, NONE, Some(3); "assignment groups")]
#[test_case(Metric::L2, BOTH, Some(2); "l2 floors groups")]
#[test_case(Metric::NegDot, Floors { row: false, col: true }, None; "neg dot col floor")]
#[test_case(Metric::NegDot, Floors { row: true, col: false }, Some(4); "neg dot row floor groups")]
fn long_lists_match_in_every_regime(metric: Metric, floors: Floors, groups: Option<i32>) {
    let bf16 = Case::draw(50, 150, 24, draw(metric, floors, groups, true, true), 9);
    let device = crate::sweep::declared(crate::atoms::sm86());
    let interpret = |prog: crate::ir::Program, _: &crate::lower::Lowering, params| run(&prog, params, &[]).unwrap();
    for case in [bf16.clone(), bf16.in_f16()] {
        for k in LONG_KS {
            let mut lists = vec![];
            for cfg in long_cfgs(k) {
                lists.push((format!("{cfg:?}"), case.listed(cfg, k)));
                if cfg.select.is_some() {
                    let blocks = case.selected(cfg, k, (17, 40), &device, &interpret);
                    lists.push((format!("{cfg:?} in blocks"), blocks));
                }
            }
            for (what, (score, index)) in &lists {
                case.check_list(&format!("{what} k{k}"), k, score, index, 1e-4, true);
                assert_eq!((score, index), (&lists[0].1.0, &lists[0].1.1), "{what} k{k}");
            }
        }
    }
}

/// Every candidate repeats one row, so every score of a row ties: its list
/// is the first `k` candidates in order, whether a trip holds a pair of them
/// or all, whatever the shares, under either sign; past the candidates `+∞`
/// at −1.
#[test_case(16, Metric::SquaredL2, 1; "a trip per pair of candidates")]
#[test_case(64, Metric::SquaredL2, 1; "every candidate in one trip")]
#[test_case(16, Metric::NegDot, 1; "signed scores a trip per pair of candidates")]
#[test_case(16, Metric::SquaredL2, 3; "three shares")]
#[test_case(16, Metric::NegDot, 4; "signed scores a share per trip")]
fn equal_scores_list_the_smallest_indices(bkv: usize, metric: Metric, splits: usize) {
    for m in [50, 20] {
        let mut case = Case::draw(30, m, 8, draw(metric, NONE, None, true, false), 2);
        for j in 1..case.m {
            case.c.copy_within(0..case.d, j * case.d);
        }
        let cfg = shared_out(NearestCfg { splits, ..NearestCfg::new(16, bkv, 2) }, m);
        let long =
            LONG_KS.into_iter().flat_map(|k| long_cfgs(k).into_iter().map(move |c| (k, NearestCfg { bkv, ..c })));
        for (k, cfg) in KS.into_iter().map(|k| (k, cfg)).chain(long) {
            let (score, index) = case.listed(cfg, k);
            let want: Vec<f64> = (0..k).map(|j| if j < m { j as f64 } else { -1.0 }).collect();
            assert!(index.chunks(k).all(|row| row == want), "m {m} k{k} {cfg:?}: {index:?}");
            case.check_list(&format!("m {m} k{k} {cfg:?}"), k, &score, &index, 0.0, true);
        }
    }
}

/// A row with fewer admissible candidates than its list fills the rest
/// with `+∞` at −1: three candidates outside every row's group, or fewer
/// candidates than slots at all; resident, streamed and split.
#[test]
fn a_list_past_the_admissible_candidates_ends_in_infinity() {
    let mut case = Case::draw(40, 50, 16, draw(Metric::L2, BOTH, Some(2), false, false), 4);
    let (rows, cols) = case.groups.as_mut().unwrap();
    rows.fill(0);
    for (j, g) in cols.iter_mut().enumerate() {
        *g = i32::from([5, 23, 47].contains(&j));
    }
    let few = Case::draw(30, 5, 16, draw(Metric::NegDot, NONE, None, false, false), 6);
    let cfgs = [
        NearestCfg::new(16, 16, 2),
        NearestCfg::streamed(32, 16, 16, 2),
        NearestCfg { splits: 3, ..NearestCfg::new(16, 16, 2) },
    ];
    let long = [33, 100].into_iter().flat_map(|k| long_cfgs(k).into_iter().map(move |c| (k, c)));
    let cfgs: Vec<(usize, NearestCfg)> =
        cfgs.into_iter().flat_map(|c| [4, 8, config::REGISTER_K].map(|k| (k, c))).chain(long).collect();
    for (case, admissible) in [(&case, 3), (&few, 5)] {
        for &(k, cfg) in &cfgs {
            {
                let (score, index) = case.listed(shared_out(cfg, case.m), k);
                case.check_list(&format!("{cfg:?} k{k}"), k, &score, &index, 1e-4, false);
                for (s, i) in score.chunks(k).zip(index.chunks(k)) {
                    let filled = admissible.min(k);
                    assert!(s[..filled].iter().all(|s| s.is_finite()) && i[..filled].iter().all(|&i| i >= 0.0));
                    assert!(s[filled..].iter().all(|s| s.is_infinite()) && i[filled..].iter().all(|&i| i == -1.0));
                }
            }
        }
    }
}

/// The merge offers the shares' lists in order: across shares an equal
/// score keeps the earlier share's (smaller) index, an empty slot (`+∞` at
/// −1) is passed over, and the scores come out in the metric's units (the
/// L2 family's doubled, L2's square root taken); rows past a block of the
/// merge are ragged.
#[test_case(Metric::NegDot; "neg dot")]
#[test_case(Metric::SquaredL2; "squared l2")]
#[test_case(Metric::L2; "l2")]
fn the_merge_offers_lists_in_order(metric: Metric) {
    let (n, splits, k, inf) = (70, 3, 4, f64::INFINITY);
    let stride = share_stride(n);
    let (mut score, mut index) = (vec![0.0; splits * k * stride], vec![0.0; splits * k * stride]);
    let mut want = vec![];
    for i in 0..n {
        // Share `s`'s candidates are `100·s..`: its list ascending, ties by index.
        let mut entries = vec![];
        for s in 0..splits {
            let filled = (i + s) % (k + 1);
            let base = 100.0 * s as f64;
            let list: Vec<(f64, f64)> = (0..k)
                .map(|j| match j < filled {
                    true => ((((i * 7 + j * 3) % 5) as f64 * 0.5 + j as f64 * 0.25).abs(), base + j as f64),
                    false => (inf, -1.0),
                })
                .collect();
            let mut list = list;
            list.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
            for (j, &(v, at)) in list.iter().enumerate() {
                let v = if metric == Metric::NegDot && v.is_finite() { v - 1.0 } else { v };
                (score[(s * k + j) * stride + i], index[(s * k + j) * stride + i]) = (v, at);
                if v.is_finite() {
                    entries.push((v, at));
                }
            }
        }
        entries.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        entries.resize(k.max(entries.len()), (inf, -1.0));
        let finish = |v: f64| match metric {
            Metric::NegDot => v,
            Metric::L2 => round_to(ScalarDType::Float32, (2.0 * v).sqrt()),
            _ => 2.0 * v,
        };
        want.extend(entries[..k].iter().map(|&(v, at)| (finish(v), at)));
    }
    let (score, index) = merged(&MergeSpec { n, splits, metric, k, live: Default::default() }, score, index);
    for (e, want) in want.into_iter().enumerate() {
        assert_eq!((score[e], index[e]), want, "row {} slot {}", e / k, e % k);
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
    let (streamed, resident, big) = (NearestCfg::streamed(128, 128, 32, 2), NearestCfg::new(128, 32, 2), 1usize << 15);
    let split = |n, m, d, cfg| planner.nearest_split(&self_join(n, m, d, Metric::SquaredL2, 1), &cfg);
    let s = split(big, big, 768, streamed).expect("a streamed self-join splits");
    assert!(s.is_power_of_two() && big.div_ceil(streamed.bkv) / s >= 4, "{s}");
    assert_eq!(split(big, big, 128, resident), None, "a resident self-join fills the SMs whole");
    assert!(split(300, big, 128, resident).is_some(), "three query blocks leave SMs idle");
    assert_eq!(split(big, 7 * 128, 768, streamed), None, "seven candidate tiles make one share of four");
    let admitted = [streamed, NearestCfg::streamed(64, 64, 32, 2), resident];
    let timed = planner.nearest_splits(&self_join(big, big, 768, Metric::SquaredL2, 1), &admitted);
    let whole: Vec<NearestCfg> = timed.iter().filter(|c| c.splits == 1).copied().collect();
    assert_eq!(whole, [streamed, resident], "{timed:?}");
    assert_eq!(timed.len(), 4, "{timed:?}");
}

/// The planner's problem of `n` rows against `m` candidates at depth `d`,
/// keeping `k` scores a row, with neither floors nor groups.
fn self_join(n: usize, m: usize, d: usize, metric: Metric, k: usize) -> config::Nearest {
    config::Nearest { n, m, d, metric, floors: NONE, groups: false, k }
}

// ---- device ------------------------------------------------------------------

/// Every config of the first eight tiles ranked for the case's shape, lowered
/// and run on the device, against the f64 reference as the interpreter is:
/// exactly where the values are integers ([non-negative](Case::unsigned)).
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
    let case = Case::draw(n, m, d, draw, 7).unsigned(draw.integer);
    let depth = planner.nearest_depth(d).unwrap();
    let ranked = planner.nearest_ranked(case.problem(depth, 1));
    let (want_score, want_index) = case.interpreted(ranked[0]);
    case.check("interpreter", &want_score, &want_index, 1e-4, draw.integer);
    for cfg in split_too(&planner, &case, depth, 1, &ranked) {
        let (score, index) = case.on_device(&planner, depth, cfg, 1);
        case.check(&format!("{cfg:?}"), &score, &index, 1e-3, draw.integer);
        let same = index.iter().zip(&want_index).filter(|(a, b)| a == b).count();
        eprintln!("{cfg:?}: {same} of {n} indices as interpreted");
    }
}

/// The first eight tiles of `ranked` (for `k` scores a row) as the tuner
/// times them (each unsplit, then at the planner's split where there is one)
/// and the first also split two and three ways.
fn split_too(planner: &Planner, case: &Case, depth: usize, k: usize, ranked: &[NearestCfg]) -> Vec<NearestCfg> {
    let mut cfgs = planner.nearest_splits(&case.problem(depth, k), &first_eight(ranked));
    let registers = ranked.iter().find(|c| c.queue.is_none() && c.select.is_none()).filter(|_| k <= 32);
    cfgs.extend(
        registers.into_iter().flat_map(|&c| [2, 3].map(|splits| shared_out(NearestCfg { splits, ..c }, case.m))),
    );
    cfgs
}

/// A resident trip finds its least's column only in the warps it lowers some
/// row of, by a vote; a streamed tile, folded once per depth pipeline, finds
/// it straight.
#[test_case(crate::sweep::amd(AmdArch::Gfx1201, 64), "@llvm.amdgcn.ballot.i32("; "gfx1201")]
#[test_case(crate::atoms::sm86(), "@llvm.nvvm.vote.any.sync("; "sm_86")]
fn a_resident_trip_votes_before_it_places_its_least(target: Target, vote: &str) {
    let device = crate::sweep::declared(target);
    let case = |cfg| NearestSpec {
        n: 300,
        m: 500,
        d: 64,
        metric: Metric::NegDot,
        floors: NONE,
        groups: false,
        k: 1,
        cfg,
        live: Default::default(),
    };
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
        let ranked = planner.nearest_ranked(case.problem(d, 1));
        for cfg in split_too(&planner, &case, d, 1, &ranked) {
            let (score, index) = case.on_device(&planner, d, cfg, 1);
            case.check(&format!("{cfg:?}"), &score, &index, 0.0, true);
        }
    }
}

/// Every config of the first eight tiles ranked for `k` scores a row (and
/// the tuner's splits), lowered and run on the device in bf16 and in f16,
/// against the reference as the interpreter is: exactly where the values are
/// integers ([non-negative](Case::unsigned)). Resident and streamed rows,
/// lists longer than the admissible candidates, exact ties among repeated
/// candidates.
#[test_case(300, 500, 64, draw(Metric::ShiftedL2, NONE, None, false, true), 4; "assignment d64 k4")]
#[test_case(257, 333, 5, draw(Metric::L2, BOTH, Some(3), false, false), 2; "l2 floors groups d5 k2")]
#[test_case(130, 270, 37, draw(Metric::L2, BOTH, Some(2), true, true), 16; "integer ties d37 k16")]
#[test_case(64, 1000, 128, draw(Metric::NegDot, Floors { row: false, col: true }, Some(4), false, false), 8; "neg dot d128 k8")]
#[test_case(100, 200, 32, draw(Metric::SquaredL2, NONE, None, true, false), config::REGISTER_K; "squared l2 d32 longest")]
#[test_case(70, 20, 16, draw(Metric::SquaredL2, NONE, Some(3), true, true), config::REGISTER_K; "fewer candidates than slots")]
#[test_case(200, 300, 512, draw(Metric::NegDot, NONE, None, false, false), 8; "streamed neg dot d512 k8")]
#[test_case(130, 170, 1000, draw(Metric::L2, BOTH, Some(3), true, true), 16; "streamed integer ties d1000 k16")]
#[test_case(150, 250, 768, draw(Metric::SquaredL2, NONE, Some(2), false, false), config::REGISTER_K; "streamed squared l2 groups d768 longest")]
#[test_case(300, 2000, 64, draw(Metric::ShiftedL2, NONE, None, false, true), 100; "assignment d64 k100")]
#[test_case(257, 1000, 5, draw(Metric::L2, BOTH, Some(3), false, false), 256; "l2 floors groups d5 k256")]
#[test_case(130, 700, 37, draw(Metric::L2, BOTH, Some(2), true, true), 64; "integer ties d37 k64")]
#[test_case(64, 3000, 128, draw(Metric::NegDot, Floors { row: false, col: true }, Some(4), false, false), 1024; "neg dot d128 k1024")]
#[test_case(70, 20, 16, draw(Metric::SquaredL2, NONE, Some(3), true, true), 33; "fewer candidates than a long list")]
#[test_case(150, 1200, 768, draw(Metric::SquaredL2, NONE, Some(2), false, false), 128; "streamed squared l2 groups d768 k128")]
fn lists_match_on_device(n: usize, m: usize, d: usize, draw: Draw, k: usize) {
    let Some(planner) = super::device_planner() else { return };
    let bf16 = Case::draw(n, m, d, draw, 7).unsigned(draw.integer);
    for case in [bf16.clone(), bf16.in_f16()] {
        let depth = planner.nearest_depth(d).unwrap();
        let ranked = planner.nearest_ranked(case.problem(depth, k));
        let (want_score, want_index) = case.listed(ranked[0], k);
        case.check_list("interpreter", k, &want_score, &want_index, 1e-4, draw.integer);
        for cfg in split_too(&planner, &case, depth, k, &ranked) {
            let what = format!("{cfg:?} k{k} f16 {}", case.f16);
            let (score, index) = case.on_device(&planner, depth, cfg, k);
            case.check_list(&what, k, &score, &index, 1e-3, draw.integer);
            let same = index.iter().zip(&want_index).filter(|(a, b)| a == b).count();
            eprintln!("{what}: {same} of {} indices as interpreted", n * k);
        }
    }
}

/// [`the_trip_that_lowers_a_row_places_its_least`] for lists, on the host
/// and on the device for every config of the first eight tiles ranked:
/// every trip offering every row a whole list's worth (the scores falling),
/// only the first trip (rising, every later trip skipped by the vote), or
/// ties everywhere; exactly the reference's lists.
#[test_case(Order::Falling; "the last candidates")]
#[test_case(Order::Rising; "the first candidates")]
#[test_case(Order::Flat; "every candidate alike")]
fn the_trips_that_lower_a_row_list_its_least(order: Order) {
    let planner = super::device_planner();
    for d in [16, 512] {
        let case = ordered(150, 230, d, order);
        for k in [2, 8, 32, 64, 100] {
            let registers =
                [NearestCfg::new(32, 16, 2), NearestCfg::new(16, 64, 1), NearestCfg::streamed(32, 32, 16, 2)];
            let registers = registers.into_iter().filter(|_| k <= config::REGISTER_K);
            let cfgs: Vec<NearestCfg> = registers.chain(if k > 32 { long_cfgs(k) } else { vec![] }).collect();
            for cfg in cfgs {
                let (score, index) = case.listed(cfg, k);
                case.check_list(&format!("{cfg:?} k{k}"), k, &score, &index, 0.0, true);
            }
            let Some(planner) = &planner else { continue };
            let ranked = planner.nearest_ranked(case.problem(d, k));
            for cfg in split_too(planner, &case, d, k, &ranked) {
                let (score, index) = case.on_device(planner, d, cfg, k);
                case.check_list(&format!("{cfg:?} k{k} on the device"), k, &score, &index, 0.0, true);
            }
        }
    }
}

/// Every lane queue width and depth lists exactly where every candidate
/// lowers its row's `k`-th (falling scores: every score is pushed, queues
/// fill and passes stop many times a trip), where every candidate ties
/// (the vote's index decides), and on random scores with groups: in
/// registers and queued into the outputs.
#[test_case(Order::Falling; "every candidate lower")]
#[test_case(Order::Flat; "every candidate alike")]
fn every_lane_queue_lists_its_least(order: Order) {
    let falling = ordered(40, 230, 16, order);
    let random = Case::draw(40, 230, 16, draw(Metric::SquaredL2, NONE, Some(3), true, true), 11);
    for width in [2, 4, 8, 16] {
        for depth in [1, 2, 4, 8] {
            let lanes = Some(LaneQueue { width, depth });
            let cfgs = [(NearestCfg::new(32, 16, 2), [3, 16, 64]), (NearestCfg::new(16, 64, 1), [1, 5, 33])];
            for (cfg, ks) in cfgs.into_iter().filter(|(c, _)| c.bkv.is_multiple_of(width)) {
                for k in ks.into_iter().filter(|&k| k > 1) {
                    let cfg = NearestCfg { lanes, ..cfg };
                    for case in [&falling, &random] {
                        let (score, index) = case.listed(cfg, k);
                        case.check_list(&format!("{cfg:?} k{k}"), k, &score, &index, 0.0, true);
                    }
                }
            }
            if 2 * width * depth <= 32 {
                let cfg = NearestCfg { lanes, queue: Some(32), ..NearestCfg::new(32, 16, 2) };
                let (score, index) = falling.listed(cfg, 100);
                falling.check_list(&format!("{cfg:?} k100"), 100, &score, &index, 0.0, true);
            }
        }
    }
}

/// A list's rounds run behind a vote whether its rows are resident or
/// streamed: the trips that lower no row's last score skip them.
#[test_case(crate::sweep::amd(AmdArch::Gfx1201, 64), "@llvm.amdgcn.ballot.i32("; "gfx1201")]
#[test_case(crate::atoms::sm86(), "@llvm.nvvm.vote.any.sync("; "sm_86")]
fn a_list_votes_before_its_rounds(target: Target, vote: &str) {
    let device = crate::sweep::declared(target);
    let case = |cfg| NearestSpec {
        n: 300,
        m: 500,
        d: 64,
        metric: Metric::NegDot,
        floors: NONE,
        groups: false,
        k: 8,
        cfg,
        live: Default::default(),
    };
    for cfg in [NearestCfg::new(64, 32, 2), NearestCfg::streamed(64, 32, 32, 2)] {
        let ir = crate::sweep::render(nearest::<BF16>(&case(cfg)), &cfg.lowering(&device));
        assert!(ir.contains(vote), "{cfg:?}");
    }
}

/// A list and its lanes' queues are held as their rows' least is, their
/// pairs' columns spread over the lanes sharing those rows
/// ([`TileLayout::spread_row`]), so no value of the kernel is re-held,
/// resident or streamed, on every target.
#[test_case(crate::atoms::sm86(); "sm_86")]
#[test_case(crate::sweep::amd(AmdArch::Gfx1201, 64); "gfx1201")]
#[test_case(crate::sweep::amd(AmdArch::Gfx1151, 40); "gfx1151")]
#[test_case(crate::sweep::amd(AmdArch::Gfx942, 304); "gfx942")]
fn a_list_is_held_beside_its_rows(target: Target) {
    use crate::ir::{Stmt, TileOp};
    use crate::layouts::TileLayout;
    let device = crate::sweep::declared(target);
    for (cfg, k) in
        [(NearestCfg::new(64, 32, 2), 8), (NearestCfg::streamed(64, 64, 32, 2), 16), (NearestCfg::new(32, 16, 2), 3)]
    {
        let spec = NearestSpec {
            n: 300,
            m: 500,
            d: 64,
            metric: Metric::SquaredL2,
            floors: NONE,
            groups: false,
            k,
            cfg,
            live: Default::default(),
        };
        let grid = crate::layouts::WarpGrid { rows: cfg.warps(), cols: 1 };
        let laid = crate::layouts::infer(nearest::<BF16>(&spec), &device, grid, Core::Warp).unwrap();
        let (mut lists, mut deals) = (0, 0);
        for (_, stmt) in laid.prog.walk() {
            let Stmt::Let { dst, op } = stmt else { continue };
            assert!(!matches!(op, TileOp::Relayout { .. }), "{cfg:?} k{k}: a relayout of {dst:?}");
            if let TileOp::Deal { src, j, .. } = *op {
                let [s, d] = [src, *dst].map(|v| laid.layouts[v.index()].as_ref().unwrap());
                let moves = s.deal(d, j, cfg.warps(), device.target.wave).unwrap();
                assert!(moves.iter().all(|(picks, _)| picks.iter().all(|p| p.mask == 0)), "{cfg:?} k{k}: deal {j}");
                deals += 1;
            }
            let w = cfg.lanes.unwrap_or(LaneQueue::PORTABLE).width;
            if laid.prog.value(*dst).shape == crate::ir::Shape::new(cfg.bq, w) {
                let l = laid.layouts[dst.index()].as_ref().unwrap();
                assert_eq!(*l, TileLayout::spread_row(&l.row_vector(), w as u32), "{cfg:?} k{k}");
                lists += 1;
            }
        }
        assert!(lists > 0 && deals > 0, "{cfg:?} k{k}: a list dealt to");
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
            _ => tk::nearest::picked(&x, &c, 1, opts, split).unwrap(),
        };
        super::ops::assert_kernel(&score, if splits == 1 { "nearest" } else { "nearest_merge" });
        case.check("kernel", &host(&score), &host(&index), 1e-3, false);
        let (score, index) = tk::nearest::graph(&x, &c, opts).unwrap();
        case.check("graph", &host(&score), &host(&index), 1e-3, false);
    });
}

/// `x`, `c` and the options of `case` as tensors of `dtype` (a self-join's
/// `c` and column floor and groups the rows' own).
fn operands(case: &Case, dtype: DType, self_join: bool) -> (Tensor, Tensor, Vec<Option<Tensor>>) {
    let f32s = |v: &[f64]| Tensor::from_slice(v.iter().map(|&a| a as f32).collect::<Vec<_>>());
    let d = case.d as isize;
    let rows = |v: &[f64], r: usize| f32s(v).cast(dtype.clone()).try_reshape([r as isize, d]).unwrap();
    let x = rows(&case.x, case.n);
    let c = if self_join { x.clone() } else { rows(&case.c, case.m) };
    let row_floor = case.row_floor.as_deref().map(f32s);
    let col_floor = if self_join { row_floor.clone() } else { case.col_floor.as_deref().map(f32s) };
    let group = |g: &[i32]| Tensor::from_slice(g);
    let (row_group, col_group) = case.groups.as_ref().map(|(r, c)| (group(r), group(c))).unzip();
    (x, c, vec![row_floor, col_floor, row_group, col_group])
}

/// `case` as the op's options over `opts` (row floor, column floor, row
/// and column groups).
fn options<'a>(case: &Case, opts: &'a [Option<Tensor>]) -> Nearest<'a> {
    let groups = opts[2].as_ref().zip(opts[3].as_ref());
    Nearest { metric: case.metric, row_floor: opts[0].as_ref(), col_floor: opts[1].as_ref(), groups }
}

/// `case` made a self-join: its candidates, column floor and groups the rows'.
fn self_joined(mut case: Case) -> Case {
    (case.c, case.m, case.col_floor) = (case.x.clone(), case.n, case.row_floor.clone());
    if let Some((r, c)) = &mut case.groups {
        *c = r.clone();
    }
    case
}

/// The op keeps each row's `k` least through the kernel, its graph keeps
/// the same lists, both against the reference: an assignment at a padded
/// depth, L2 over a self-join under floors and groups, the inner product
/// under groups, embeddings' depths whose rows stream, in bf16 and f16 and
/// split; `k = 1` is the op's least as a column.
#[test_case(300, 500, 5, draw(Metric::ShiftedL2, NONE, None, false, false), false, 1, 4; "assignment d5 k4")]
#[test_case(300, 300, 16, draw(Metric::L2, BOTH, Some(5), false, false), true, 1, 8; "l2 floors groups self-join k8")]
#[test_case(100, 130, 128, draw(Metric::NegDot, NONE, Some(3), false, false), false, 1, 16; "neg dot groups d128 k16")]
#[test_case(500, 700, 768, draw(Metric::NegDot, NONE, None, false, false), false, 1, 32; "streamed neg dot d768 longest")]
#[test_case(300, 300, 1000, draw(Metric::SquaredL2, BOTH, Some(4), false, false), true, 1, 1; "streamed self-join d1000 k1")]
#[test_case(300, 300, 16, draw(Metric::L2, BOTH, Some(5), false, false), true, 3, 8; "l2 floors groups self-join split k8")]
#[test_case(500, 700, 768, draw(Metric::SquaredL2, NONE, None, false, false), false, 2, 16; "streamed squared l2 d768 split k16")]
#[test_case(300, 300, 16, draw(Metric::L2, BOTH, Some(5), false, false), true, 1, 64; "l2 floors groups self-join k64")]
#[test_case(200, 900, 128, draw(Metric::NegDot, NONE, Some(3), false, false), false, 1, 100; "neg dot groups d128 k100")]
#[test_case(300, 230_000, 16, draw(Metric::NegDot, NONE, None, false, false), false, 1, 64; "neg dot past a score block k64")]
fn lists_match_the_graph(n: usize, m: usize, d: usize, draw: Draw, self_join: bool, splits: usize, k: usize) {
    super::ops::untuned(|| {
        for f16 in [false, true] {
            let case = Case::draw(n, m, d, draw, 11);
            let case = if self_join { self_joined(case) } else { case };
            let case = if f16 { case.in_f16() } else { case };
            let dtype = if f16 { DType::Float16 } else { DType::BFloat16 };
            let (x, c, opts) = operands(&case, dtype, self_join);
            let opts = options(&case, &opts);
            let split = |_: &Planner, ranked: &[NearestCfg], spec: NearestSpec| {
                let cfg = shared_out(NearestCfg { splits, ..ranked[0] }, spec.m);
                NearestSpec { cfg, ..spec }
            };
            let (score, index) = match splits {
                1 => tk::k_nearest(&x, &c, k, opts).unwrap(),
                _ => tk::nearest::picked(&x, &c, k, opts, split).unwrap(),
            };
            let rows = if splits == 1 && k == 1 { vec![n, 1] } else { vec![n, k] };
            assert_eq!(score.dims().unwrap(), rows, "{f16}");
            super::ops::assert_kernel(&score, if splits == 1 { "nearest" } else { "nearest_merge" });
            case.check_list(&format!("kernel f16 {f16}"), k, &host(&score), &host(&index), 1e-3, false);
            // A long list both ways: queued, and through a score matrix.
            for matrix in (k > config::REGISTER_K).then_some([false, true]).into_iter().flatten() {
                // The way's best ranked config, or where the ranking prunes it, its first admitted.
                let way = |planner: &Planner, ranked: &[NearestCfg], spec: NearestSpec| {
                    let p = tk::nearest::problem(&spec);
                    let lattice = p.lattice(&planner.device.target);
                    let admitted = lattice.iter().filter(|c| p.admit(planner, c).is_ok());
                    let mut ways = ranked.iter().chain(admitted).filter(|c| c.select.is_some() == matrix);
                    NearestSpec { cfg: *ways.next().expect("a way of a long list"), ..spec }
                };
                let (score, index) = tk::nearest::picked(&x, &c, k, opts, way).unwrap();
                super::ops::assert_kernel(&score, if matrix { "nearest_select" } else { "nearest" });
                case.check_list(&format!("matrix {matrix} f16 {f16}"), k, &host(&score), &host(&index), 1e-3, false);
            }
            let (score, index) = tk::nearest::graph_k(&x, &c, k, opts).unwrap();
            case.check_list(&format!("graph f16 {f16}"), k, &host(&score), &host(&index), 1e-3, false);
        }
    });
}

/// What the kernel does not serve takes the graph and keeps the same lists,
/// on any device: f32 operands, a list longer than a register list holds,
/// one longer than the candidates; `k = 1` as a column of the least.
#[test_case(DType::Float32, 8, 90; "f32")]
#[test_case(DType::Float32, 1, 90; "f32 least alone")]
#[test_case(DType::Float32, config::REGISTER_K + 1, 90; "an f32 list past the registers")]
#[test_case(DType::BFloat16, 12, 5; "past the candidates")]
fn a_list_the_kernel_does_not_serve_takes_the_graph(dtype: DType, k: usize, m: usize) {
    let case = self_joined(Case::draw(m, m, 24, draw(Metric::SquaredL2, BOTH, Some(3), true, true), 3));
    let (x, c, opts) = operands(&case, dtype, true);
    let (score, index) = tk::k_nearest(&x, &c, k, options(&case, &opts)).unwrap();
    assert_eq!(score.dims().unwrap(), [m, k]);
    case.check_list("graph", k, &host(&score), &host(&index), 1e-4, true);
}

/// A list of no scores is an error, not an empty tensor.
#[test]
fn a_list_of_no_scores_is_an_error() {
    let x = Tensor::from_slice(vec![0.0f32; 8]).try_reshape([2, 4]).unwrap();
    assert!(tk::k_nearest(&x, &x, 0, Nearest::default()).is_err());
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
    time_list(planner, config::Nearest { n, m: n, d, metric, floors, groups, k: 1 }, DType::BFloat16);
}

/// [`time_nearest`] of the self-join `p` (its `k` scores a row) in `dtype`.
fn time_list(planner: &Planner, p: config::Nearest, dtype: DType) {
    let config::Nearest { n, d, metric, floors, groups, k, .. } = p;
    let (admitted, pick) = device_nearest(planner, p, dtype.clone());
    let mut timed = admitted.clone();
    if !timed.contains(&pick) {
        timed.push(pick);
    }
    let spec = |cfg| NearestSpec { n, m: n, d, metric, floors, groups, k, cfg, live: Default::default() };
    let ns = super::measure(timed.iter().map(|&cfg| tk::nearest::programs(planner, dtype.clone(), &spec(cfg))));
    eprintln!(
        "== nearest {n}x{n}x{d} k{k} {dtype:?} {metric:?} {floors:?} groups={groups}: {} admitted",
        admitted.len()
    );
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

/// A row's `k` least over a self-join, per `KNN_ROWS` (comma-separated,
/// 16384,32768), `KNN_DEPTHS` (128,256,768) and `KNN_KS` (1,32,64,128,256,1024),
/// the inner product and squared distances in f16: the tuner's pick for
/// each, as the op tunes it, its pairs a second and their share of `k = 1`'s;
/// with `KNN_ALL` set, every config the tuner admitted too; with
/// `KNN_GRAPH` set, the graph fallback's at up to that many rows and
/// `KNN_GRAPH_K` (128) scores a row, beside; with `KNN_CANDIDATES` set,
/// that many candidates against the rows instead of a self-join.
/// Prints, never asserts (`--ignored --nocapture --test-threads=1`).
#[test]
#[ignore = "perf probe: needs a GPU"]
fn knn_probe() {
    let list = |name: &str, default: &str| -> Vec<usize> {
        let text = std::env::var(name).unwrap_or(default.into());
        text.split(',').map(|v| v.trim().parse().expect("a comma-separated list of counts")).collect()
    };
    let (rows, depths, ks) =
        (list("KNN_ROWS", "16384,32768"), list("KNN_DEPTHS", "128,256,768"), list("KNN_KS", "1,32,64,128,256,1024"));
    let graph_rows: usize = std::env::var("KNN_GRAPH").map_or(0, |v| v.parse().expect("KNN_GRAPH is a row count"));
    let graph_k: usize = std::env::var("KNN_GRAPH_K").map_or(128, |v| v.parse().expect("KNN_GRAPH_K is a list length"));
    let candidates: Option<usize> = std::env::var("KNN_CANDIDATES").ok().map(|v| v.parse().expect("a candidate count"));
    let Some(planner) = super::device_planner() else { return };
    crate::tune::scoped(true, || {
        for &n in &rows {
            for &d in &depths {
                for metric in [Metric::NegDot, Metric::SquaredL2] {
                    let mut least = None;
                    for &k in &ks {
                        let m = candidates.unwrap_or(n);
                        let p = config::Nearest { n, m, d, metric, floors: NONE, groups: false, k };
                        if std::env::var_os("KNN_ALL").is_some() && m == n {
                            time_list(&planner, p, DType::Float16);
                        }
                        let (_, pick) = device_nearest(&planner, p, DType::Float16);
                        let spec = NearestSpec {
                            n,
                            m,
                            d,
                            metric,
                            floors: NONE,
                            groups: false,
                            k,
                            cfg: pick,
                            live: Default::default(),
                        };
                        let [ns] = super::measure([tk::nearest::programs(&planner, DType::Float16, &spec)])[..] else {
                            unreachable!("one candidate")
                        };
                        let rate = ns.map(|ns| (n * m) as f64 / ns as f64);
                        least = least.or(rate.filter(|_| k == 1));
                        let share = rate.zip(least).map_or("-".into(), |(r, l)| format!("{:5.1} %", 100.0 * r / l));
                        let rate = rate.map_or("failed".into(), |r| format!("{r:7.1} Gpairs/s"));
                        eprintln!("knn {n}x{m} d{d} {metric:?} k{k}: {rate} {share} of k=1, {pick:?}");
                        if n <= graph_rows && k <= graph_k && m == n {
                            let graph = graph_list_seconds(n, d, metric, k)
                                .map_or("failed".into(), |s| format!("{:7.1} Gpairs/s", (n * n) as f64 / s / 1e9));
                            eprintln!("knn {n}² d{d} {metric:?} k{k} graph: {graph}");
                        }
                    }
                }
            }
        }
    });
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

/// The configs the tuner admits for the self-join `p` on the device in
/// `dtype`, and its pick.
fn device_nearest(planner: &Planner, p: config::Nearest, dtype: DType) -> (Vec<NearestCfg>, NearestCfg) {
    let config::Nearest { n, m, d, metric, floors, groups, k } = p;
    let ranked = planner.nearest_ranked(p);
    let spec = |cfg| NearestSpec { n, m, d, metric, floors, groups, k, cfg, live: Default::default() };
    let build = |cfg: NearestCfg| tk::nearest::programs(planner, dtype.clone(), &spec(cfg));
    let shape = [n, m, d, k];
    let shape = if k == 1 { &shape[..3] } else { &shape[..] };
    let (admitted, pick) = super::tuned_on_device(
        planner,
        ("nearest", dtype.clone(), shape, (crate::ops::DEVICE_TEST, metric, floors, groups)),
        &ranked,
        |admitted| planner.nearest_splits(&p, admitted),
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

/// The best of a few runs of the graph fallback's `k` least of an `n`-row
/// self-join at depth `d` in f16, in seconds.
fn graph_list_seconds(n: usize, d: usize, metric: Metric, k: usize) -> Option<f64> {
    let mut seed = 5;
    let data: Vec<f32> = (0..n * d).map(|_| super::lcg(&mut seed) as f32).collect();
    let x = Tensor::from_slice(&data).cast(DType::Float16).try_reshape([n as isize, d as isize]).ok()?.contiguous();
    x.realize().ok()?;
    let (_, index) = tk::nearest::graph_k(&x, &x, k, Nearest { metric, ..Nearest::default() }).ok()?;
    let plan = index.prepare().ok()?;
    let mut best = f64::INFINITY;
    for _ in 0..3 {
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
    let spec = NearestSpec {
        n: 1000,
        m: 1000,
        d: 64,
        metric: Metric::SquaredL2,
        floors: NONE,
        groups: false,
        k: 1,
        cfg,
        live: Default::default(),
    };
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
/// `TUNE_DEPTHS` (256,512,768,1024) keeping `TUNE_KS` (1) scores a row, the GEMMs `TUNE_GEMMS` (`MxNxK,…`,
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
    let ks: Vec<usize> = list("TUNE_KS", "1").concat();
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
        let ordered = tuner.ordered(op, &admitted, crate::tune::Pass::Screen);
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
        for (d, k) in depths.iter().flat_map(|&d| ks.iter().map(move |&k| (d, k))) {
            let (n, metric, floors) = (rows, Metric::NegDot, NONE);
            let p = Problem { n, m: n, d, metric, floors, groups: false, k };
            let ranked = planner.nearest_ranked(p);
            let spec =
                |cfg| NearestSpec { n, m: n, d, metric, floors, groups: false, k, cfg, live: Default::default() };
            let build = |cfg: NearestCfg| tk::nearest::programs(&planner, DType::BFloat16, &spec(cfg));
            let salt = (crate::ops::DEVICE_TEST, metric, floors, false);
            let name = format!("nearest {n}x{n}x{d} k{k} {metric:?}");
            let work = 2.0 * (n * n * d) as f64;
            let shape = [n, n, d, k];
            report(
                &planner,
                name,
                ("nearest", if k == 1 { &shape[..3] } else { &shape[..] }, salt),
                (&ranked, |a: &[NearestCfg]| planner.nearest_splits(&p, a)),
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

/// The 16-bit product against f64 on f32 vectors: per file of
/// `NEAREST_VECTORS` (comma-separated; `u32 n, u32 d`, then `n·d` f32
/// rows, a `<stem>.labels` of `n` i32 beside it optional), dtype (bf16,
/// f16) and metric (NegDot, SquaredL2), the first `NEAREST_QUERIES` rows
/// (2000) against the rest and the first `NEAREST_SELF` rows (20000)
/// against themselves, one group a row. Per case: rows whose index is not
/// f64's (smallest on ties); for those, f64's gap from the true neighbour
/// (absolute and over the row's score deviation), the returned candidate's
/// rank, how many lie within the first-order rounding bound of the two
/// pairs, and, labelled, how many change the neighbour's label; the
/// returned score's error against f64. `NEAREST_DUMP` writes each case's
/// indices there (`<stem>_<case>_<dtype>_<metric>.i32`); untuned unless
/// `NEAREST_TUNED` is set. Prints, never
/// asserts (`--ignored --nocapture --test-threads=1`).
#[test]
#[ignore = "accuracy probe: needs a GPU and NEAREST_VECTORS"]
fn nearest_precision_probe() {
    use rayon::prelude::*;
    let var = |name: &str| std::env::var(name).ok();
    let Some(files) = var("NEAREST_VECTORS") else { return eprintln!("skipped: NEAREST_VECTORS unset") };
    let queries: usize = var("NEAREST_QUERIES").map_or(2000, |v| v.parse().expect("NEAREST_QUERIES is a row count"));
    let self_rows: usize = var("NEAREST_SELF").map_or(20000, |v| v.parse().expect("NEAREST_SELF is a row count"));
    let dump = var("NEAREST_DUMP");
    let tuned = var("NEAREST_TUNED").is_some();
    if super::device_planner().is_none() {
        return;
    }
    for path in files.split(',') {
        let bytes = std::fs::read(path).expect("NEAREST_VECTORS names a readable file");
        let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()) as usize;
        let (rows, d) = (word(0), word(1));
        let data: Vec<f32> = (0..rows * d).map(|i| f32::from_bits(word(2 + i) as u32)).collect();
        let stem = path.strip_suffix(".bin").unwrap_or(path);
        let labels: Option<Vec<i32>> = std::fs::read(format!("{stem}.labels"))
            .ok()
            .map(|b| b.as_chunks::<4>().0.iter().map(|&w| i32::from_le_bytes(w)).collect());
        let name = std::path::Path::new(stem).file_name().unwrap().to_string_lossy().into_owned();
        let q = queries.min(rows / 2);
        let s = self_rows.min(rows);
        for (case, xs, cs, own) in [("query", 0..q, q..rows, false), ("self", 0..s, 0..s, true)] {
            let (n, m) = (xs.len(), cs.len());
            let x64: Vec<f64> = data[xs.start * d..xs.end * d].iter().map(|&v| v as f64).collect();
            let c64: Vec<f64> = data[cs.start * d..cs.end * d].iter().map(|&v| v as f64).collect();
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
            let sq = |v: &[f64]| (0..v.len() / d).map(|i| dot(&v[i * d..][..d], &v[i * d..][..d])).collect::<Vec<_>>();
            let (x_sq, c_sq) = (sq(&x64), sq(&c64));
            let score = |metric: Metric, i: usize, j: usize| {
                let p = dot(&x64[i * d..][..d], &c64[j * d..][..d]);
                if metric == Metric::NegDot { -p } else { x_sq[i] + c_sq[j] - 2.0 * p }
            };
            let tensor = |r: std::ops::Range<usize>| {
                let t = Tensor::from_slice(&data[r.start * d..r.end * d]);
                t.try_reshape([r.len() as isize, d as isize]).unwrap()
            };
            let (xt, ct) = (tensor(xs.clone()), tensor(cs.clone()));
            let ids = Tensor::from_slice((0..n as i32).collect::<Vec<_>>());
            for metric in [Metric::NegDot, Metric::SquaredL2] {
                // f64's least per row, smallest index on ties; the row's score deviation.
                let reference: Vec<(usize, f64, f64)> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let (mut best, mut at, mut sum, mut sum_sq) = (f64::INFINITY, 0, 0.0, 0.0);
                        for j in (0..m).filter(|&j| !(own && j == i)) {
                            let v = score(metric, i, j);
                            (sum, sum_sq) = (sum + v, sum_sq + v * v);
                            if v < best {
                                (best, at) = (v, j);
                            }
                        }
                        let k = (m - usize::from(own)) as f64;
                        (at, best, (sum_sq / k - (sum / k).powi(2)).max(0.0).sqrt())
                    })
                    .collect();
                for (dtype, scalar, u) in [
                    (DType::BFloat16, ScalarDType::BFloat16, 2f64.powi(-8)),
                    (DType::Float16, ScalarDType::Float16, 2f64.powi(-11)),
                ] {
                    let (x, c) = (xt.cast(dtype.clone()), if own { None } else { Some(ct.cast(dtype.clone())) });
                    let c = c.as_ref().unwrap_or(&x);
                    if case == "query" && metric == Metric::NegDot {
                        let cast = host(&x);
                        let off = (0..n * d).filter(|&k| cast[k] != round_to(scalar, x64[k])).count();
                        eprintln!("{name} {dtype:?} cast: {off} of {} elements off round-to-nearest-even", n * d);
                    }
                    let opts = Nearest { metric, groups: own.then_some((&ids, &ids)), ..Nearest::default() };
                    let (got_score, got_index) = crate::tune::scoped(tuned, || tk::nearest(&x, c, opts).unwrap());
                    let (got_score, got_index) = (host(&got_score), host(&got_index));
                    if let Some(dir) = &dump {
                        let out: Vec<u8> = got_index.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect();
                        std::fs::write(format!("{dir}/{name}_{case}_{dtype:?}_{metric:?}.i32"), out).unwrap();
                    }
                    // The first-order bound on a pair's rounding error: inputs `u` each,
                    // f32 sums over `d` terms and the epilogue's few roundings.
                    let bound = |i: usize, j: usize| {
                        let (a, b) = (&x64[i * d..][..d], &c64[j * d..][..d]);
                        let abs = a.iter().zip(b).map(|(a, b)| (a * b).abs()).sum::<f64>();
                        let e = (2.0 * u + u * u + d as f64 * 2f64.powi(-24)) * abs;
                        match metric {
                            Metric::NegDot => e,
                            _ => {
                                2.0 * e + (2.0 * u + u * u) * c_sq[j] + 2f64.powi(-22) * (x_sq[i] + c_sq[j] + 2.0 * abs)
                            }
                        }
                    };
                    let (mut wrong, mut within, mut flips) = (vec![], 0, 0);
                    let mut err: Vec<f64> = vec![];
                    for i in 0..n {
                        let (at, best, dev) = reference[i];
                        let j = got_index[i] as usize;
                        err.push((got_score[i] - score(metric, i, j)).abs());
                        if j == at {
                            continue;
                        }
                        let gap = score(metric, i, j) - best;
                        within += usize::from(gap <= bound(i, j) + bound(i, at));
                        if let Some(l) = &labels {
                            flips += usize::from(l[cs.start + j] != l[cs.start + at]);
                        }
                        wrong.push((i, gap, gap / dev.max(f64::MIN_POSITIVE)));
                    }
                    let ranks: Vec<usize> = wrong
                        .par_iter()
                        .map(|&(i, ..)| {
                            let s = score(metric, i, got_index[i] as usize);
                            1 + (0..m).filter(|&j| !(own && j == i) && score(metric, i, j) < s).count()
                        })
                        .collect();
                    let stat = |mut v: Vec<f64>| {
                        v.sort_by(f64::total_cmp);
                        match v.len() {
                            0 => "-".to_string(),
                            k => format!("{:.2e}/{:.2e}/{:.2e}", v[k / 2], v[k * 99 / 100], v[k - 1]),
                        }
                    };
                    let rms = (err.iter().map(|e| e * e).sum::<f64>() / n as f64).sqrt();
                    let labelled = labels.as_ref().map_or(String::new(), |_| format!(" label flips {flips}"));
                    eprintln!(
                        "{name} d={d} {case} {n}x{m} {dtype:?} {metric:?}: wrong {} ({:.3} %), within bound {within}{labelled}; \
                         gap p50/p99/max {} rel {}; rank {}; score err rms {rms:.2e} max {:.2e}",
                        wrong.len(),
                        100.0 * wrong.len() as f64 / n as f64,
                        stat(wrong.iter().map(|w| w.1).collect()),
                        stat(wrong.iter().map(|w| w.2).collect()),
                        stat(ranks.into_iter().map(|r| r as f64).collect()),
                        err.iter().copied().fold(0.0, f64::max),
                    );
                }
            }
        }
    }
}

mod live;

/// Writes the LLVM IR of list kernels to `LIST_IR` (a directory), one
/// `<k>_<cfg index>.ll` per `LIST_KS` (16) list and config of
/// `LIST_CFGS` (`bq:bkv:stages:depth:width[:queue]`, semicolon-separated,
/// their rings unrolled under `LIST_UNROLL`),
/// for `LIST_TARGET` (sm_86 or gfx1201): what a register count is read
/// from. Prints nothing else.
#[test]
#[ignore = "IR dump: set LIST_IR"]
fn list_ir_dump() {
    let Ok(dir) = std::env::var("LIST_IR") else { return };
    let target = match std::env::var("LIST_TARGET").as_deref() {
        Ok("gfx1201") => crate::sweep::amd(AmdArch::Gfx1201, 64),
        _ => crate::atoms::sm86(),
    };
    let device = crate::sweep::declared(target);
    let ks: Vec<usize> =
        std::env::var("LIST_KS").unwrap_or("16".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let cfgs = std::env::var("LIST_CFGS").unwrap_or("64:64:2:2:4".into());
    for (i, text) in cfgs.split(';').enumerate() {
        let v: Vec<usize> = text.split(':').map(|v| v.parse().unwrap()).collect();
        let lanes = Some(LaneQueue { width: v[4], depth: v[3] });
        let unroll = std::env::var_os("LIST_UNROLL").is_some();
        let cfg = NearestCfg { lanes, unroll, queue: v.get(5).copied(), ..NearestCfg::new(v[0], v[1], v[2]) };
        for &k in &ks {
            let spec = NearestSpec {
                n: 1 << 14,
                m: 1 << 14,
                d: 128,
                metric: Metric::SquaredL2,
                floors: NONE,
                groups: false,
                k,
                cfg,
                live: Default::default(),
            };
            let ir = crate::sweep::render(nearest::<F16>(&spec), &cfg.lowering(&device));
            std::fs::write(format!("{dir}/{k}_{i}.ll"), ir).unwrap();
        }
    }
}

/// Times list configs untuned on the device: `LIST_CFGS` as
/// [`list_ir_dump`] reads them, at `LIST_KS` (16) scores a row,
/// `LIST_ROWS` (16384) rows self-joined at `LIST_DEPTH` (128), f16
/// squared distances; `LIST_RUNS` (3) runs each, alternating. Prints, never
/// asserts (`--ignored --nocapture --test-threads=1`).
#[test]
#[ignore = "perf probe: needs a GPU"]
fn list_time_probe() {
    let Some(planner) = super::device_planner() else { return };
    let var = |name: &str, default: &str| std::env::var(name).unwrap_or(default.into());
    let ks: Vec<usize> = var("LIST_KS", "16").split(',').map(|v| v.parse().unwrap()).collect();
    let (n, d): (usize, usize) =
        (var("LIST_ROWS", "16384").parse().unwrap(), var("LIST_DEPTH", "128").parse().unwrap());
    let runs: usize = var("LIST_RUNS", "3").parse().unwrap();
    let cfgs: Vec<NearestCfg> = var("LIST_CFGS", "64:32:3:2:4")
        .split(';')
        .map(|text| {
            let v: Vec<usize> = text.split(':').map(|v| v.parse().unwrap()).collect();
            let lanes = Some(LaneQueue { width: v[4], depth: v[3] });
            let unroll = std::env::var_os("LIST_UNROLL").is_some();
            NearestCfg { lanes, unroll, queue: v.get(5).copied(), ..NearestCfg::new(v[0], v[1], v[2]) }
        })
        .collect();
    for &k in &ks {
        let spec = |cfg| NearestSpec {
            n,
            m: n,
            d,
            metric: Metric::SquaredL2,
            floors: NONE,
            groups: false,
            k,
            cfg,
            live: Default::default(),
        };
        for run in 0..runs {
            let ns =
                super::measure(cfgs.iter().map(|&cfg| tk::nearest::programs(&planner, DType::Float16, &spec(cfg))));
            for (cfg, ns) in cfgs.iter().zip(ns) {
                let rate = ns.map_or("failed".into(), |ns| format!("{:7.1} Gpairs/s", (n * n) as f64 / ns as f64));
                eprintln!("list k{k} d{d} run {run}: {rate} {cfg:?}");
            }
        }
    }
}
