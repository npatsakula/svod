//! Live rows and lists: kernels built for the bound of a symbolic row count
//! and of a list's length take the values bound at launch. Against the
//! kernels built for those values (the interpreter and the device, every
//! way of keeping a list, the edges of both bounds), and the op over
//! symbolic rows and lists, one plan rebound, against the op at each shape.

use svod_dtype::DType;
use svod_ir::SInt;
use svod_tensor::{BoundVariable, Tensor, Variable};
use test_case::test_case;

use super::*;
use crate::atoms::Device;
use crate::ir::{Program, Var};
use crate::kernels::nearest::Live;
use crate::lower::Lowering;

/// The variables the tests bind the rows and a list's length to.
const ROWS: &str = "n";
const LIST: &str = "k";

/// Runs a program on its lowering under the variables given, every
/// parameter's values after it in order.
type Exec<'a> = &'a dyn Fn(Program, &Lowering, Vec<Vec<f64>>, &[(&str, i64)]) -> Vec<Vec<f64>>;

fn interpret(prog: Program, _: &Lowering, params: Vec<Vec<f64>>, vars: &[(&str, i64)]) -> Vec<Vec<f64>> {
    run(&prog, params, vars).unwrap()
}

fn on_device(prog: Program, lowering: &Lowering, params: Vec<Vec<f64>>, vars: &[(&str, i64)]) -> Vec<Vec<f64>> {
    let outs = super::super::targets::on_device_with(prog, lowering, &params, vars);
    outs.into_iter().zip(params).map(|(o, p)| o.unwrap_or(p)).collect()
}

/// The live rows of `case` (all of them its bound) where `rows`, and lists
/// bound to `kmax` where it holds more than one score.
fn live(case: &Case, rows: bool, kmax: usize) -> Live {
    Live {
        rows: rows.then(|| (Var { name: ROWS.into(), min: 1, max: case.n as i64 }, 0)),
        k: (kmax > 1).then(|| Var { name: LIST.into(), min: 1, max: kmax as i64 }),
    }
}

/// `cfg` made a way that keeps a list of `k`, as the concrete kernel of a
/// live length `k` is: unchanged where it keeps one, else a register list
/// up to [`config::REGISTER_K`], a queue past it.
fn concrete_way(cfg: NearestCfg, k: usize) -> NearestCfg {
    let keeps = match (cfg.queue, cfg.select) {
        (Some(q), _) => q < k,
        (_, Some(s)) => s.fits(k),
        _ => k <= config::REGISTER_K,
    };
    match keeps {
        true => cfg,
        false if k <= config::REGISTER_K => NearestCfg { queue: None, select: None, ..cfg },
        false => NearestCfg { queue: Some(16), select: None, ..cfg },
    }
}

/// The ways of keeping a list of `kmax` the tests take: a register list
/// resident, streamed and split three ways (up to [`config::REGISTER_K`]),
/// queued, queued and split, and through a score matrix (from 33 on).
fn ways(kmax: usize, m: usize) -> Vec<NearestCfg> {
    let mut out = vec![];
    if kmax <= config::REGISTER_K {
        let split = shared_out(NearestCfg { splits: 3, ..NearestCfg::new(32, 16, 2) }, m);
        out.extend([NearestCfg::new(32, 16, 2), NearestCfg::streamed(32, 32, 16, 1), split]);
    }
    if kmax >= 33 {
        let long = long_cfgs(kmax);
        out.push(shared_out(NearestCfg { splits: 3, ..long[0] }, m));
        out.extend(long);
    }
    out
}

impl Case {
    fn compiled(&self, spec: &NearestSpec) -> Program {
        if self.f16 { nearest::<F16>(spec) } else { nearest::<BF16>(spec) }
    }

    /// The lists of the first `n` rows, `k` slots each, as the kernels built
    /// for all the case's rows and for lists of `kmax` under `cfg` leave
    /// them with `live` bound to `n` and `k` (rows not live are all `n`, a
    /// list not live is `kmax` long); a score matrix's in panels of `panel`
    /// rows and blocks of `cols` candidates. Run by `exec`.
    #[allow(clippy::too_many_arguments)]
    fn live_listed(
        &self,
        cfg: NearestCfg,
        live: &Live,
        kmax: usize,
        (n, k): (usize, usize),
        (panel, cols): (usize, usize),
        device: &Device,
        exec: Exec,
    ) -> (Vec<f64>, Vec<f64>) {
        assert!((live.rows.is_some() || n == self.n) && (live.k.is_some() || k == kmax), "unbound values");
        let depth = self.d.next_multiple_of(16);
        let rows_var = live.rows.iter().map(|_| (ROWS, n as i64));
        let vars: Vec<(&str, i64)> = rows_var.chain(live.k.iter().map(|_| (LIST, k as i64))).collect();
        let (score, index) = match cfg.select {
            Some(select) => {
                // Filtered, its bound read at the live length.
                let select = SelectCfg { filter: true, ..select };
                let cfg = NearestCfg { select: Some(select), ..cfg };
                let (scored, listing) = (Lowering::new(device), Lowering::new(device));
                let (mut score, mut index) = (vec![], vec![]);
                for r0 in (0..self.n).step_by(panel) {
                    let rows = panel.min(self.n - r0);
                    let live = Live { rows: live.rows.clone().map(|(v, _)| (v, r0)), k: live.k.clone() };
                    let mut lists = vec![vec![0.0; rows * kmax]; 2];
                    for c0 in (0..self.m).step_by(cols) {
                        let width = cols.min(self.m - c0);
                        // Live, every panel reads the rows whole.
                        let held = if live.rows.is_some() { 0..self.n } else { r0..r0 + rows };
                        let part = self.part(held, c0..c0 + width);
                        let mut params = part.params(depth, 1, kmax);
                        params.truncate(params.len() - 2);
                        let groups = width.div_ceil(cfg.bkv);
                        params.extend([vec![0.0; rows * width], vec![0.0; rows * groups]]);
                        let spec = NearestSpec { n: rows, m: width, live: live.clone(), ..part.spec(depth, cfg, kmax) };
                        let mut out = exec(part.compiled(&spec), &scored, params, &vars);
                        let (mins, matrix) = (out.pop().unwrap(), out.pop().unwrap());
                        let (metric, first) = (self.metric, c0 == 0);
                        let spec = |cols, first, filter, shares| SelectSpec {
                            n: rows,
                            cols,
                            k: kmax,
                            metric,
                            first,
                            cfg: select,
                            live: live.clone(),
                            wide: select.wide(device.target.wave),
                            filter,
                            shares,
                        };
                        // Filtered by the live `k`-th of the trips' leasts.
                        let lists_out = vec![vec![0.0; rows * kmax]; 2];
                        let params = [vec![mins.clone(), vec![0.0; 2]], lists_out].concat();
                        let mut out = exec(nearest_select(&spec(groups, true, None, 1)), &listing, params, &vars);
                        out.truncate(out.len() - 1);
                        let filter = vec![mins, out.pop().unwrap()];
                        // Whole rows list in two shares, merged at the live length.
                        let shares = if width == self.m { spec(width, true, None, 1).steps().min(2) } else { 1 };
                        let spec = spec(width, first, Some(cfg.bkv), shares);
                        let chunk = vec![c0 as f64, f64::from(c0 + width == self.m)];
                        let carried = if first { vec![] } else { lists };
                        let parts = kmax * if shares > 1 { shares * share_stride(rows) } else { rows };
                        let params = [vec![matrix, chunk], carried, filter, vec![vec![0.0; parts]; 2]].concat();
                        let mut out = exec(nearest_select(&spec), &listing, params, &vars);
                        lists = out.split_off(out.len() - 2);
                        if shares > 1 {
                            let merge =
                                NearestMergeSpec { n: rows, splits: shares, metric, k: kmax, live: live.clone() };
                            let params = [lists, vec![vec![0.0; rows * kmax]; 2]].concat();
                            let mut out = exec(nearest_merge(&merge), &Lowering::new(device), params, &vars);
                            lists = out.split_off(out.len() - 2);
                        }
                    }
                    score.extend(&lists[0]);
                    index.extend(&lists[1]);
                }
                (score, index)
            }
            None => {
                let spec = NearestSpec { live: live.clone(), ..self.spec(depth, cfg, kmax) };
                let params = self.params(depth, cfg.splits, kmax);
                let mut out = exec(self.compiled(&spec), &Lowering::new(device), params, &vars);
                let (index, score) = (out.pop().unwrap(), out.pop().unwrap());
                if cfg.splits == 1 {
                    (score, index)
                } else {
                    let merge = NearestMergeSpec { live: live.clone(), ..self.merge(cfg.splits, kmax) };
                    let params = vec![score, index, vec![0.0; self.n * kmax], vec![0.0; self.n * kmax]];
                    let mut out = exec(nearest_merge(&merge), &Lowering::new(device), params, &vars);
                    let index = out.pop().unwrap();
                    (out.pop().unwrap(), index)
                }
            }
        };
        let cut = |v: Vec<f64>| (0..n).flat_map(|i| v[i * kmax..][..k].to_vec()).collect::<Vec<_>>();
        (cut(score), cut(index))
    }

    /// The case's first `n` rows against every candidate.
    fn first(&self, n: usize) -> Case {
        self.part(0..n, 0..self.m)
    }
}

/// The interpreter's lists of `case` under `cfg` built for its rows and
/// lists of `kmax` with `live` bound to `(n, k)`, bit for bit those of the
/// kernel built for `n` rows and lists of `k` (under [`concrete_way`]), and
/// the reference's.
fn assert_live_interpreted(
    case: &Case,
    cfg: NearestCfg,
    live: &Live,
    kmax: usize,
    nk: (usize, usize),
    panels: (usize, usize),
) {
    let device = crate::sweep::declared(crate::atoms::sm86());
    let (n, k) = nk;
    let what = format!("{cfg:?} {live:?} bound to n {n} k {k} of {} rows, {kmax} slots", case.n);
    let got = case.live_listed(cfg, live, kmax, nk, panels, &device, &interpret);
    let small = case.first(n);
    let want = small.listed(concrete_way(cfg, k), k);
    assert_eq!(got, want, "{what}");
    small.check_list(&what, k, &got.0, &got.1, 1e-4, false);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]
    /// Rows and lists bound below their bounds, or at them, keep the lists
    /// the kernels built for the bound values keep, bit for bit, in every
    /// way of keeping a list: over any shape, every metric, either floor,
    /// with and without groups, exact ties, lists longer than a row's
    /// admissible candidates, panels of a score matrix past the bound rows
    /// (launching none of their own), in bf16 and f16.
    #[test]
    fn live_lists_keep_the_lists_of_their_values(
        cap in 1usize..48,
        m in 1usize..120,
        d in 1usize..40,
        draw in draws(),
        kmax in prop::sample::select(vec![1, 2, 7, 16, 33, 64, 100, 256]),
        way in 0usize..8,
        rows in any::<bool>(),
        list in any::<bool>(),
        at in (0.0f64..1.0, 0.0f64..1.0),
        panels in (1usize..48, 1usize..150),
        f16 in any::<bool>(),
        seed in any::<u64>(),
    ) {
        let case = Case::draw(cap, m, d, draw, seed);
        let case = if f16 { case.in_f16() } else { case };
        let ways = ways(kmax, m);
        let cfg = ways[way % ways.len()];
        let n = if rows { 1 + ((cap - 1) as f64 * at.0) as usize } else { cap };
        let k = if list && kmax > 1 { 1 + ((kmax - 1) as f64 * at.1) as usize } else { kmax };
        let live = live(&case, rows, if list { kmax } else { 1 });
        assert_live_interpreted(&case, cfg, &live, kmax, (n, k), panels);
    }
}

/// Each way at the edges of its bounds: one row, every row, a list of one,
/// of its bound, and longer than the candidates, rows and lists live alone
/// and together; a score matrix in panels of 17 rows, so one bound row
/// leaves two panels empty.
#[test_case(NearestCfg::new(32, 16, 2), 1; "a least alone")]
#[test_case(NearestCfg { splits: 3, ..NearestCfg::new(32, 16, 2) }, 1; "a split least alone")]
#[test_case(NearestCfg::new(32, 16, 2), 16; "registers")]
#[test_case(NearestCfg::streamed(32, 32, 16, 1), config::REGISTER_K; "streamed registers")]
#[test_case(NearestCfg { splits: 3, ..NearestCfg::new(32, 16, 2) }, 8; "split registers")]
#[test_case(NearestCfg { splits: 3, ..NearestCfg::new(32, 16, 2) }, config::REGISTER_K; "split registers longest")]
#[test_case(NearestCfg { queue: Some(16), ..NearestCfg::new(32, 16, 2) }, 100; "queued")]
#[test_case(NearestCfg { queue: Some(16), splits: 3, ..NearestCfg::new(32, 16, 2) }, 100; "split queued")]
#[test_case(NearestCfg { queue: Some(32), ..NearestCfg::streamed(32, 32, 16, 1) }, 256; "streamed queued")]
#[test_case(NearestCfg { queue: Some(32), splits: 4, ..NearestCfg::streamed(32, 32, 16, 1) }, 256; "split streamed queued")]
#[test_case(NearestCfg { select: Some(SelectCfg { warps: 2, queue: 2, filter: false }), ..NearestCfg::new(16, 16, 2) }, 100; "score matrix")]
#[test_case(NearestCfg { select: Some(SelectCfg { warps: 4, queue: 8, filter: false }), ..NearestCfg::new(64, 64, 1) }, 1024; "score matrix 1024")]
fn live_bounds_keep_their_edges(cfg: NearestCfg, kmax: usize) {
    let cap = 45;
    for m in [150, 20] {
        let case = Case::draw(cap, m, 24, draw(Metric::L2, BOTH, Some(3), true, true), 7);
        let cfg = shared_out(cfg, m);
        for (rows, list) in [(true, false), (false, true), (true, true)] {
            for n in if rows { vec![1, 17, cap] } else { vec![cap] } {
                let ks = if list && kmax > 1 { vec![1, 2, kmax / 2, kmax - 1, kmax] } else { vec![kmax] };
                for k in ks {
                    let live = live(&case, rows, if list { kmax } else { 1 });
                    assert_live_interpreted(&case, cfg, &live, kmax, (n, k), (17, 64));
                }
            }
        }
    }
}

/// The device keeps each way's live lists bit for bit as it keeps the lists
/// of their values (the same products, folded alike), at a few bindings of
/// both bounds: the planner's first config of each way for the bound, and a
/// queued list's past the registers where the ranking pruned them.
#[test_case(16, 64; "registers d64")]
#[test_case(64, 768; "registers streamed d768")]
#[test_case(8, 128; "split registers d128")]
#[test_case(100, 128; "split queued and score matrix k100")]
#[test_case(256, 768; "queued and score matrix k256 d768")]
fn live_lists_keep_the_lists_of_their_values_on_device(kmax: usize, d: usize) {
    let Some(planner) = super::super::device_planner() else { return };
    let (cap, m) = (300, 1100);
    let case = Case::draw(cap, m, d, draw(Metric::SquaredL2, NONE, Some(3), false, false), 5).in_f16();
    let depth = d.next_multiple_of(16);
    let problem = case.problem(depth, kmax);
    let ranked = planner.nearest_ranked(problem);
    // Split three ways where the way splits (a score matrix never does).
    let splits = |c: &NearestCfg| match (kmax, c.select) {
        (8 | 100, None) => shared_out(NearestCfg { splits: 3, ..*c }, m),
        _ => *c,
    };
    let queued = (kmax > config::REGISTER_K)
        .then(|| {
            problem
                .lattice(&planner.device.target)
                .into_iter()
                .find(|c| c.queue.is_some() && problem.admit(&planner, c).is_ok())
        })
        .flatten();
    let mut kinds = vec![];
    for cfg in ranked.iter().copied().chain(queued).map(|c| splits(&c)) {
        let kind = (cfg.queue.is_some(), cfg.select.is_some(), cfg.splits > 1);
        if !kinds.contains(&kind) {
            kinds.push(kind);
            for (n, k) in [(1, 1), (37, kmax.div_ceil(3)), (cap, kmax), (cap - 1, kmax - 1)] {
                let (n, k) = (n, k.max(1));
                let live = live(&case, true, kmax);
                let what = format!("{cfg:?} n {n} k {k}");
                let blocks = (cap.div_ceil(2), m.div_ceil(3));
                let got = case.live_listed(cfg, &live, kmax, (n, k), blocks, &planner.device, &on_device);
                let small = case.first(n);
                let want = small.on_device(&planner, depth, concrete_way(cfg, k), k);
                assert_eq!(got, want, "{what}");
                small.check_list(&what, k, &got.0, &got.1, 1e-2, false);
            }
        }
    }
    assert!(kmax == 8 || kinds.len() > 1 || kmax <= config::REGISTER_K, "k {kmax}: ways {kinds:?}");
}

// ---- the op -----------------------------------------------------------------

/// `t`'s leading dim bound to `var`, its capacity the rest.
fn bound(t: &Tensor, var: &BoundVariable) -> Tensor {
    let mut ranges: Vec<Option<(SInt, SInt)>> = vec![None; t.shape().unwrap().len()];
    ranges[0] = Some((SInt::Const(0), var.as_sint()));
    t.try_shrink(ranges).unwrap()
}

/// The prepared plans of `outs` (f32 and contiguous), whose variables each
/// run binds anew. A realized symbolic dim is laid out at its bound, as
/// Tinygrad's: a `[N, k]` output's rows are `k`'s bound apart.
struct Rebound(Vec<svod_runtime::ExecutionPlan>);

impl Rebound {
    fn new(outs: &[&Tensor]) -> Self {
        Self(outs.iter().map(|t| t.cast(DType::Float32).contiguous().prepare().unwrap()).collect())
    }

    /// Every output's first `rows` rows of `cols` values, `stride` apart,
    /// under `vars`.
    fn run(&mut self, vars: &[(&str, i64)], (rows, cols, stride): (usize, usize, usize)) -> Vec<Vec<f64>> {
        self.0
            .iter_mut()
            .map(|plan| {
                plan.execute_with_vars(vars).unwrap();
                let mut bytes = vec![0u8; rows * stride * 4];
                plan.output_buffer().unwrap().copyout_prefix(&mut bytes).unwrap();
                let values: Vec<f64> =
                    bytes.chunks(4).map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap()))).collect();
                (0..rows).flat_map(|i| values[i * stride..][..cols].to_vec()).collect()
            })
            .collect()
    }
}

/// `case`'s operands with the rows (`x`, the row floor and groups) bound to
/// `var` where given: `(x, c, options)`.
fn bound_operands(case: &Case, dtype: DType, var: Option<&BoundVariable>) -> (Tensor, Tensor, Vec<Option<Tensor>>) {
    let (x, c, mut opts) = operands(case, dtype, false);
    let Some(var) = var else { return (x, c, opts) };
    for at in [0, 2] {
        opts[at] = opts[at].as_ref().map(|t| bound(t, var));
    }
    (bound(&x, var), c, opts)
}

/// The op over `case` with rows bound up to its rows and a list's length
/// bound up to `kmax`, through the way `pick` makes of the planned spec:
/// one plan, rebound to each `(n, k)`, keeps the lists the op keeps at
/// that shape through the same config, bit for bit; the outputs `[N, k]`.
/// Returns the config the kernel ran, if one did.
fn assert_live_op(
    case: &Case,
    dtype: DType,
    kmax: usize,
    bindings: &[(usize, usize)],
    pick: &dyn Fn(&Planner, &[NearestCfg], NearestSpec) -> NearestSpec,
) -> Option<NearestCfg> {
    let n_var = Variable::new(ROWS, 1, case.n as i64).bind(bindings[0].0 as i64).unwrap();
    let k_var = Variable::new(LIST, 1, kmax as i64).bind(bindings[0].1 as i64).unwrap();
    let (x, c, opts) = bound_operands(case, dtype.clone(), Some(&n_var));
    let mut used = None;
    let live_pick = |planner: &Planner, ranked: &[NearestCfg], spec: NearestSpec| {
        let spec = pick(planner, ranked, spec);
        used = Some(spec.cfg);
        spec
    };
    let (score, index) = tk::nearest::picked(&x, &c, k_var.as_sint(), options(case, &opts), None, live_pick).unwrap();
    assert_eq!(score.shape().unwrap().to_vec(), [n_var.as_sint(), k_var.as_sint()]);
    assert_eq!(index.shape().unwrap().to_vec(), [n_var.as_sint(), k_var.as_sint()]);
    if let Some(cfg) = used {
        let kernel = match (cfg.select, cfg.splits) {
            (Some(_), _) => "nearest_select",
            (None, 1) => "nearest",
            _ => "nearest_merge",
        };
        super::super::ops::assert_kernel(&score, kernel);
    }
    let mut plans = Rebound::new(&[&score, &index]);
    for &(n, k) in bindings {
        let got = plans.run(&[(ROWS, n as i64), (LIST, k as i64)], (n, k, kmax));
        let small = case.first(n);
        let (x, c, opts) = bound_operands(&small, dtype.clone(), None);
        let same = |_: &Planner, _: &[NearestCfg], spec: NearestSpec| match used {
            Some(cfg) => NearestSpec { cfg: concrete_way(cfg, k), ..spec },
            None => spec,
        };
        let (score, index) = tk::nearest::picked(&x, &c, k, options(&small, &opts), None, same).unwrap();
        let rows = |t: &Tensor| t.try_reshape([(n * k) as isize]).unwrap();
        let want = [host(&rows(&score)), host(&rows(&index))];
        assert_eq!(got, want, "n {n} k {k} under {used:?}");
        // The reference of many rows is the host's minutes; the op's own lists are checked elsewhere.
        if n * case.m <= 1 << 22 {
            small.check_list(&format!("n {n} k {k}"), k, &got[0], &got[1], 1e-2, false);
        }
    }
    used
}

/// The planner's first config of a way, else its best admitted (the
/// ranking prunes all but the cheapest products of a score matrix).
fn way(queue: bool, select: bool, splits: usize) -> impl Fn(&Planner, &[NearestCfg], NearestSpec) -> NearestSpec {
    move |planner, ranked, spec| {
        let p = tk::nearest::problem(&spec, svod_dtype::ScalarDType::Float16);
        let lattice = p.lattice(&planner.device.target);
        let admitted = lattice.iter().filter(|c| p.admit(planner, c).is_ok());
        let mut ways =
            ranked.iter().chain(admitted).filter(|c| c.queue.is_some() == queue && c.select.is_some() == select);
        let cfg = *ways.next().expect("the way admitted");
        NearestSpec { cfg: shared_out(NearestCfg { splits, ..cfg }, spec.m), ..spec }
    }
}

/// The op over live rows and lists on the device, each way: a least alone
/// (`k` bound to 1), register lists resident, streamed and split, queued and
/// through a score matrix; one plan rebound from one row to all of them,
/// from a list of one to its bound, past the candidates.
#[test_case(1, 64, false, false, 1, &[(300, 1), (1, 1), (151, 1)]; "a least alone")]
#[test_case(16, 64, false, false, 1, &[(300, 16), (1, 1), (151, 5), (299, 16)]; "registers")]
#[test_case(64, 768, false, false, 1, &[(300, 64), (1, 64), (77, 63), (300, 1)]; "registers streamed")]
#[test_case(8, 128, false, false, 3, &[(300, 8), (17, 3), (300, 1)]; "split registers")]
#[test_case(100, 128, true, false, 1, &[(300, 100), (1, 33), (151, 99), (300, 2)]; "queued")]
#[test_case(100, 128, true, false, 4, &[(300, 100), (1, 33), (151, 99), (300, 2)]; "split queued")]
#[test_case(256, 64, false, true, 1, &[(300, 256), (2, 1), (151, 100), (300, 255)]; "score matrix")]
fn live_op_keeps_the_op_lists(
    kmax: usize,
    d: usize,
    queue: bool,
    select: bool,
    splits: usize,
    bindings: &[(usize, usize)],
) {
    super::super::ops::untuned(|| {
        let case = Case::draw(300, 1000, d, draw(Metric::NegDot, BOTH, Some(4), false, false), 13);
        let cfg = assert_live_op(&case, DType::Float16, kmax, bindings, &way(queue, select, splits)).expect("a kernel");
        assert_eq!((cfg.queue.is_some(), cfg.select.is_some(), cfg.splits), (queue, select, splits), "{cfg:?}");
        // Past the candidates: the slots past them `+∞` at −1.
        let few = Case::draw(300, 20, d, draw(Metric::NegDot, BOTH, Some(4), false, false), 13);
        assert_live_op(&few, DType::Float16, kmax, &[(300, kmax), (7, kmax)], &way(queue, select, splits));
    });
}

/// The reranked op over live rows and lists: one plan rebound from the
/// bounds to fewer rows and shorter lists keeps each row's exact `k` least
/// at every binding, the outputs `[N, k]`.
#[test_case(16, 128, Metric::SquaredL2, &[(300, 16), (1, 1), (151, 5)]; "squared l2")]
#[test_case(64, 768, Metric::NegDot, &[(300, 64), (77, 63), (300, 1)]; "streamed neg dot")]
fn live_reranked_op_keeps_exact_lists(kmax: usize, d: usize, metric: Metric, bindings: &[(usize, usize)]) {
    super::super::ops::untuned(|| {
        let case = Case::draw(300, 1000, d, draw(metric, NONE, Some(4), false, false), 17);
        let n_var = Variable::new(ROWS, 1, case.n as i64).bind(bindings[0].0 as i64).unwrap();
        let k_var = Variable::new(LIST, 1, kmax as i64).bind(bindings[0].1 as i64).unwrap();
        let (x, c, opts) = bound_operands(&case, DType::Float32, Some(&n_var));
        let rerank = tk::Rerank::default();
        let (score, index) = tk::k_nearest_reranked(&x, &c, k_var.as_sint(), options(&case, &opts), rerank).unwrap();
        assert_eq!(score.shape().unwrap().to_vec(), [n_var.as_sint(), k_var.as_sint()]);
        super::super::ops::assert_kernel(&score, "nearest_select");
        let mut plans = Rebound::new(&[&score, &index]);
        for &(n, k) in bindings {
            let got = plans.run(&[(ROWS, n as i64), (LIST, k as i64)], (n, k, kmax));
            case.first(n).check_list(&format!("n {n} k {k}"), k, &got[0], &got[1], 1e-5, true);
        }
    });
}

/// Live rows over a score matrix too large for one block of rows: every
/// panel takes the bound rows past its first, the last none of them, one,
/// or all its own.
#[test]
fn live_rows_span_the_panels_of_a_score_matrix() {
    super::super::ops::untuned(|| {
        let (cap, m, kmax) = (140_000, 2100, 256);
        let case = Case::draw(cap, m, 16, draw(Metric::NegDot, NONE, None, false, false), 21).in_f16();
        let pick = way(false, true, 1);
        let panels = |planner: &Planner, ranked: &[NearestCfg], spec: NearestSpec| {
            let spec = pick(planner, ranked, spec);
            // Panels where an eighth of the device's memory is short of the
            // matrix (12 GB and less), whole rows past it.
            let scratch = crate::kernels::nearest::scratch(planner.device.target.memory_bytes);
            let (panel, _) = crate::kernels::nearest::select_blocks(spec.n, spec.m, spec.k, &spec.cfg, scratch);
            let paneled = spec.n * spec.m > scratch;
            assert!(panel > 1 && (panel < cap) == paneled, "{panel} rows a panel of {cap} under {scratch}");
            spec
        };
        let bindings = [(cap, kmax), (100, kmax), (131_073, 7), (cap - 1, 1)];
        assert!(assert_live_op(&case, DType::Float16, kmax, &bindings, &panels).is_some_and(|c| c.select.is_some()));
    });
}

/// The graph keeps live rows and lists on any device (f32 takes it): one
/// plan rebound to each shape keeps the graph's lists at that shape.
#[test_case(1, &[(40, 1), (1, 1), (23, 1)]; "a least alone")]
#[test_case(8, &[(40, 8), (1, 1), (23, 5)]; "a list")]
#[test_case(12, &[(40, 12), (9, 12)]; "past the candidates")]
fn live_graph_keeps_the_graph_lists(kmax: usize, bindings: &[(usize, usize)]) {
    let m = if kmax == 12 { 5 } else { 50 };
    let case = Case::draw(40, m, 24, draw(Metric::SquaredL2, BOTH, Some(3), true, true), 3);
    let keep = |_: &Planner, _: &[NearestCfg], spec: NearestSpec| spec;
    assert_eq!(assert_live_op(&case, DType::Float32, kmax, bindings, &keep), None, "f32 takes the graph");
}

/// A least alone over live rows: `[N]` outputs, as [`tk::nearest`]'s.
#[test]
fn a_least_over_live_rows_keeps_the_least() {
    let case = Case::draw(40, 50, 24, draw(Metric::L2, BOTH, Some(3), false, false), 3);
    for dtype in [DType::Float32, DType::Float16] {
        let var = Variable::new(ROWS, 1, 40).bind(40).unwrap();
        let (x, c, opts) = bound_operands(&case, dtype.clone(), Some(&var));
        let (score, index) = tk::nearest(&x, &c, options(&case, &opts)).unwrap();
        assert_eq!(score.shape().unwrap().to_vec(), [var.as_sint()]);
        let mut plans = Rebound::new(&[&score, &index]);
        for n in [40, 1, 17] {
            let got = plans.run(&[(ROWS, n as i64)], (n, 1, 1));
            case.first(n).check(&format!("{dtype:?} n {n}"), &got[0], &got[1], 1e-2, false);
        }
    }
}

/// A list's length is a count or a bound variable, at least one.
#[test]
fn a_list_length_is_a_count_or_a_bound_variable() {
    let x = Tensor::from_slice(vec![0.0f32; 8]).try_reshape([2, 4]).unwrap();
    let k = Variable::new(LIST, 1, 4).bind(2).unwrap();
    let expr = k.as_sint() - 1usize;
    assert!(matches!(tk::k_nearest(&x, &x, expr, Nearest::default()), Err(tk::Error::Unbound { .. })));
    let none = Variable::new(LIST, 0, 4).bind(1).unwrap();
    assert!(matches!(tk::k_nearest(&x, &x, none.as_sint(), Nearest::default()), Err(tk::Error::NoScores { .. })));
}

/// Rows quantized beforehand for a reranked search are those of the tensor
/// they stand in for: another shape is an error before any device is asked.
#[test]
fn prequantized_rows_of_another_shape_are_an_error() {
    let x = Tensor::from_slice(vec![0.0f32; 8]).try_reshape([2, 4]).unwrap();
    let other = Tensor::from_slice(vec![0.0f32; 12]).try_reshape([3, 4]).unwrap();
    let q = tk::Quantized::new(&other).unwrap();
    for (x8, c8, operand) in [(Some(&q), None, "x8"), (None, Some(&q), "c8")] {
        let rerank = tk::Rerank { x8, c8, ..tk::Rerank::default() };
        let got = tk::k_nearest_reranked(&x, &x, 1usize, Nearest::default(), rerank);
        assert!(matches!(got, Err(tk::Error::Shape { operand: o, .. }) if o == operand), "{operand}");
    }
    let q = tk::Quantized::new(&x).unwrap();
    let rerank = tk::Rerank { x8: Some(&q), c8: Some(&q), ..tk::Rerank::default() };
    assert!(tk::k_nearest_reranked(&x, &x, 1usize, Nearest::default(), rerank).is_ok());
}

/// Rows that may bind to none (a variable down to 0): the kernel and the
/// graph (f32) keep the lists of the rows bound. (Bound to none, the graph's
/// own kernels launch an empty grid, which the runtime refuses.)
#[test_case(DType::Float16; "kernel")]
#[test_case(DType::Float32; "graph")]
fn rows_that_may_be_none(dtype: DType) {
    let case = Case::draw(40, 50, 24, draw(Metric::NegDot, NONE, None, false, false), 3);
    let var = Variable::new(ROWS, 0, 40).bind(5).unwrap();
    let (x, c, opts) = bound_operands(&case, dtype, Some(&var));
    let (score, index) = tk::k_nearest(&x, &c, 4, options(&case, &opts)).unwrap();
    let mut plans = Rebound::new(&[&score, &index]);
    for n in [5, 1, 40] {
        let got = plans.run(&[(ROWS, n as i64)], (n, 4, 4));
        case.first(n).check_list(&format!("n {n}"), 4, &got[0], &got[1], 1e-2, false);
    }
}

/// A live tuning keys apart from the tuning of its bound shape (the kernels
/// differ) and alike at every binding (no bound value is in it); a concrete
/// one keys as it did before live rows existed.
#[test]
fn live_tunings_key_apart_from_their_bounds() {
    use crate::ops::nearest::Salt;
    let spec = |live| NearestSpec {
        n: 300,
        m: 500,
        d: 64,
        metric: Metric::L2,
        floors: BOTH,
        groups: true,
        k: 16,
        cfg: NearestCfg::new(32, 16, 2),
        live,
    };
    let concrete = spec(Live::default());
    assert_eq!(format!("{:?}", Salt(&concrete)), format!("{:?}", (Metric::L2, BOTH, true)));
    let case = Case::draw(300, 500, 64, draw(Metric::L2, BOTH, Some(3), false, false), 1);
    let salts: Vec<String> =
        [(true, 1), (false, 16), (true, 16)].map(|(rows, k)| format!("{:?}", Salt(&spec(live(&case, rows, k))))).into();
    for (i, salt) in salts.iter().enumerate() {
        assert_ne!(*salt, format!("{:?}", Salt(&concrete)));
        assert!(salts[..i].iter().all(|s| s != salt), "{salts:?}");
    }
}

/// The tuner's pick of the op's spec, kept in `cfg`.
fn tuned_into(
    cfg: &std::cell::Cell<Option<NearestCfg>>,
    dtype: DType,
) -> impl FnOnce(&Planner, &[NearestCfg], NearestSpec) -> NearestSpec + '_ {
    move |planner, ranked, spec| {
        let spec = tk::nearest::tuned(crate::tune::TuneStore::global(), planner, dtype, ranked, spec);
        cfg.set(Some(spec.cfg));
        spec
    }
}

/// The op's run time over live rows against the op at the bound rows'
/// count, f16, `LIVE_ROWS` (16384) rows against as many candidates (their
/// values, another buffer) at `LIVE_DEPTHS`
/// (128, 768), lists of `LIVE_KS` (1, 16, 64, 256), every kernel of a run
/// summed, best of `LIVE_RUNS` (5) runs, the two alternated `LIVE_ROUNDS`
/// (3) times: the concrete op (its tuned pick), the live op bound at the
/// same count under the concrete pick's config (what the live rows cost)
/// and under its own tuned pick, and rows bound at twice as many run at
/// the count (one plan for both). The live rows are a realized buffer, as
/// a JIT's symbolic batch is (a view of a static one is copied first).
/// Prints, never asserts
/// (`--ignored --nocapture --test-threads=1`).
#[test]
#[ignore = "perf probe: needs a GPU"]
fn live_probe() {
    let list = |name: &str, default: &str| -> Vec<usize> {
        let text = std::env::var(name).unwrap_or(default.into());
        text.split(',').map(|v| v.trim().parse().expect("a comma-separated list of counts")).collect()
    };
    let (n, depths, ks) =
        (list("LIVE_ROWS", "16384")[0], list("LIVE_DEPTHS", "128,768"), list("LIVE_KS", "1,16,64,256"));
    let (runs, rounds) = (list("LIVE_RUNS", "5")[0], list("LIVE_ROUNDS", "3")[0]);
    let Some(_) = super::super::device_planner() else { return };
    let dtype = DType::Float16;
    let timed = |plan: &mut svod_runtime::ExecutionPlan, vars: &[(&str, i64)]| {
        plan.execute_with_vars(vars).unwrap();
        (0..runs).filter_map(|_| crate::tune::run(plan)).min().map_or(f64::NAN, |t| t.as_secs_f64() * 1e3)
    };
    crate::tune::scoped(true, || {
        for &d in &depths {
            for &k in &ks {
                let mut seed = 5;
                let data: Vec<f32> = (0..2 * n * d).map(|_| super::super::lcg(&mut seed) as f32).collect();
                let all = Tensor::from_slice(&data).cast(dtype.clone()).try_reshape([(2 * n) as isize, d as isize]);
                let all = all.unwrap().contiguous();
                all.realize().unwrap();
                let x = all.narrow(0, 0, n).unwrap().contiguous();
                // The candidates a buffer of their own, the same values as
                // the rows: a self-join of one tensor would read one buffer
                // for both, which the live rows (a buffer of their own)
                // cannot.
                let c = all.narrow(0, 0, n).unwrap().contiguous();
                x.realize().unwrap();
                c.realize().unwrap();
                let opts = Nearest { metric: Metric::SquaredL2, ..Nearest::default() };
                let tuned = |cfg| tuned_into(cfg, dtype.clone());
                // The concrete op and its pick.
                let pick = std::cell::Cell::new(None);
                let (_, index) = tk::nearest::picked(&x, &c, k, opts, None, tuned(&pick)).unwrap();
                let mut concrete = index.prepare().unwrap();
                // Rows bound at the count, under the concrete pick.
                let at = Variable::new(ROWS, 1, n as i64).bind(n as i64).unwrap();
                let same = |_: &Planner, _: &[NearestCfg], spec: NearestSpec| match pick.get() {
                    Some(cfg) => NearestSpec { cfg, ..spec },
                    None => spec,
                };
                // Realized as the JIT hands a symbolic batch on: a view of a
                // static buffer would be copied first.
                let xb = bound(&x, &at).contiguous();
                xb.realize().unwrap();
                let (_, index) = tk::nearest::picked(&xb, &c, k, opts, None, same).unwrap();
                let mut live_same = index.prepare().unwrap();
                // Rows bound at the count under their own pick, and at twice it.
                let own = std::cell::Cell::new(None);
                let (_, index) = tk::nearest::picked(&xb, &c, k, opts, None, tuned(&own)).unwrap();
                let mut live_own = index.prepare().unwrap();
                let twice = Variable::new(ROWS, 1, 2 * n as i64).bind(n as i64).unwrap();
                let wide = std::cell::Cell::new(None);
                let xw = bound(&all, &twice).contiguous();
                xw.realize().unwrap();
                let (_, index) = tk::nearest::picked(&xw, &c, k, opts, None, tuned(&wide)).unwrap();
                let mut live_wide = index.prepare().unwrap();
                let vars = [(ROWS, n as i64)];
                let mut best = [f64::INFINITY; 4];
                for _ in 0..rounds {
                    let times = [
                        timed(&mut concrete, &[]),
                        timed(&mut live_same, &vars),
                        timed(&mut live_own, &vars),
                        timed(&mut live_wide, &vars),
                    ];
                    for (b, t) in best.iter_mut().zip(times) {
                        *b = b.min(t);
                    }
                }
                let [c, s, o, w] = best;
                let share = |t: f64| format!("{t:8.3} ms {:+6.1} %", 100.0 * (t / c - 1.0));
                eprintln!(
                    "live {n}x{n} d{d} k{k}: concrete {c:8.3} ms {:?} | live same cfg {} | live own {} {:?} | bound {} run at {n} {} {:?}",
                    pick.get(),
                    share(s),
                    share(o),
                    own.get(),
                    2 * n,
                    share(w),
                    wide.get(),
                );
            }
        }
    });
}
