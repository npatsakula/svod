//! The rank harness's own math and replay: Spearman over ties, and a
//! synthetic store replayed through the very keys the op layer writes —
//! the digest assert holding, a drifted list refused and counted.

use test_case::test_case;

use super::Scratch;
use svod_dtype::ScalarDType;

use crate::atoms::sm86;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec};
use crate::ops::config::{GemmProblem, Planner, Tiled, representatives};
use crate::ops::tuned;
use crate::rank;
use crate::tune::{Entry, Kind, TuneKey, TuneStore};

/// Spearman's ρ of a perfect order, an inverted one, ties, and a side with
/// no variance at all.
#[test_case(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0], Some(1.0); "a perfect order")]
#[test_case(&[1.0, 2.0, 3.0], &[30.0, 20.0, 10.0], Some(-1.0); "an inverted order")]
#[test_case(&[1.0, 1.0, 2.0], &[1.0, 2.0, 3.0], Some(1.5 / (1.5_f64 * 2.0).sqrt()); "ties averaged")]
#[test_case(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0], None; "one side constant")]
#[test_case(&[5.0], &[1.0], None; "one candidate")]
fn spearman_correlates(a: &[f64], b: &[f64], want: Option<f64>) {
    match (rank::spearman(a, b), want) {
        (Some(got), Some(want)) => assert!((got - want).abs() < 1e-12, "{got} against {want}"),
        (got, want) => assert_eq!(got, want),
    }
}

/// A store under `dir` holding a 512³ GEMM's admission — the first four of
/// its ranking's representatives — and its screened choice, each candidate
/// at the ns `ns` of its predicted cost, plus every probe value line at the
/// guessed costs (so the harness's planner ranks under the costs it reads
/// back). The lines are keyed as `linear`'s tuning keys them. The planner
/// the store makes.
fn store(dir: &std::path::Path, ns: impl Fn(usize) -> u64) -> Planner {
    let target = sm86();
    let file = TuneStore::at(Some(dir.to_path_buf())).file("sm_86-28sm").expect("a file");
    let mut text = format!("format {}\n", crate::tune::FORMAT);
    let mut guessed = crate::probe::GUESSED;
    for (name, value) in guessed.fields() {
        text += &format!("value|probe:{name}|sm_86-28sm|Float16|0|0|0|0 {}\n", (*value * 1e6) as u64);
    }
    std::fs::write(&file, &text).expect("written");
    let measured = crate::probe::stored(dir, &target).expect("the file's costs");
    let planner = Planner::measured(target.clone(), measured).expect("an SM count");
    let ranked = planner.gemm_ranked(1, 512, 512, 512, false);
    let reps = representatives(&ranked);
    let salt = format!("{:?}", (Batch::Static(1), Epilogue::DEFAULT));
    let keyed = |rule: &str, list: &[GemmCfg]| {
        let candidates = format!("{salt}{rule} {list:?}");
        TuneKey::new("linear", &target, 0, ScalarDType::BFloat16, &[1, 512, 512, 512], &candidates)
    };
    let at: Vec<usize> = (0..4).collect();
    let admitted: Vec<GemmCfg> = at.iter().map(|&i| reps[i]).collect();
    let mut timed: Vec<(usize, u64)> = (0..4).map(|i| (i, ns(i))).collect();
    timed.sort_by_key(|&(_, ns)| ns);
    let at: Vec<(usize, String)> = at.iter().map(|&i| (i, reps[i].code())).collect();
    let timed: Vec<(usize, String, u64)> = timed.into_iter().map(|(i, ns)| (i, admitted[i].code(), ns)).collect();
    text += &format!("{} {}\n", keyed("", &reps).line(Kind::Admitted, 1), Entry::Admitted(at.into()).render());
    text +=
        &format!("{} {}\n", keyed(" screened", &admitted).line(Kind::Choice, 1), Entry::Choice(timed.into()).render());
    std::fs::write(&file, &text).expect("written");
    planner
}

/// A 512³ GEMM's first four representatives — the store's candidates — and
/// the ns the model says they cost.
fn predicted(planner: &Planner) -> Vec<f64> {
    let ranked = planner.gemm_ranked(1, 512, 512, 512, false);
    let p = GemmProblem { batches: 1, m: 512, n: 512, k: 512, halves: 1 };
    representatives(&ranked).iter().take(4).map(|&c| p.admit(planner, &c).expect("ranked").0).collect()
}

/// A store the op layer's own keys name replays: the digest asserts the
/// candidate list, and the line correlates as its ns order says — perfectly
/// where the device agreed with the model, negatively where it inverted it.
#[test_case(|c: f64| (c * 3.0 + 17.0) as u64; "the device agreed with the model")]
#[test_case(|c: f64| (9.0e5 - c * 3.0) as u64; "the device inverted the model")]
fn a_store_replays_through_the_op_layers_keys(ns: fn(f64) -> u64) {
    let agreed = (ns(1.0) as i64 - ns(0.0) as i64).signum() > 0;
    let name = if agreed { "rank-agree" } else { "rank-invert" };
    let order = Scratch::new(&format!("{name}-order"));
    let p = predicted(&store(&order, |at| at as u64));
    assert!(p.windows(2).all(|w| w[0] <= w[1]) && p[0] < p[3], "the model's order varies: {p:?}");
    let dir = Scratch::new(name);
    store(&dir, |at| ns(p[at]));
    let devices = rank::replay(&dir, None, Some(&["gemm"]));
    assert_eq!(devices.len(), 1, "the one device file");
    let family = &devices[0].families[0];
    assert_eq!((family.name, family.lines.len(), family.skipped), ("gemm", 1, 0));
    let rho = family.lines[0].rho(&devices[0].planner).expect("the costs give the order variance");
    assert!(if agreed { rho > 0.99 } else { rho < -0.99 }, "{rho}");
    assert_eq!(family.lines[0].top1(&devices[0].planner), agreed);
}

/// A choice line whose candidates digest no ranking of the shape makes — a
/// model that moved, a list the code at hand no longer derives — replays
/// from the configs its own codes carry, counted stored beside the derived
/// line; a line whose codes no hypothesis of the shape admits (an unknown
/// salt) is skipped and counted, never guessed at.
#[test]
fn a_line_no_ranking_makes_replays_from_its_own_configs() {
    let dir = Scratch::new("rank-drift");
    let planner = store(&dir, |at| 100 + at as u64);
    let file = TuneStore::at(Some(dir.to_path_buf())).file("sm_86-28sm").expect("a file");
    let text = std::fs::read_to_string(&file).expect("read");
    let ranked = planner.gemm_ranked(1, 512, 512, 512, false);
    let drifted: Vec<GemmCfg> = representatives(&ranked).into_iter().skip(1).take(4).collect();
    let made = |key: u64| {
        format!(
            "choice|linear|sm_86-28sm|BFloat16|1x512x512x512|{key:016x}|0|0 {}\n",
            (0..drifted.len())
                .map(|i| format!("{}:{} {}", i, drifted[i].code(), 400 + 10 * i))
                .collect::<Vec<_>>()
                .join(" ")
        )
    };
    std::fs::write(&file, format!("{text}{}{}", made(0xdeadbeefdeadbeef), made(0xbeefbeefbeefbeef))).expect("written");
    // The drifted digest is not the one the ranking derives; the line still
    // replays from its codes, in their ns order. Its codes are the ranking's
    // own configs, so the hypothesis admits every one.
    let devices = rank::replay(&dir, None, Some(&["gemm"]));
    let family = &devices[0].families[0];
    assert_eq!((family.lines.len(), family.stored, family.skipped), (2, 1, 0), "derived and stored lines");
    let stored = family.lines.iter().find(|l| l.rule == " stored").expect("the stored line");
    let rho = stored.rho(&devices[0].planner).expect("variance");
    assert!(rho > 0.9, "its ns order is its predicted order: {rho}");
}

/// The nearest epilogue's weight seam: each cluster scales only the
/// constants of the work it names — a row's least (`k = 1`) holds none of
/// the listing, merge or scan work, so zeroing those leaves its cost the
/// same bits, while a list config's and a score matrix's cost move with
/// theirs.
#[test]
fn the_epilogue_clusters_scale_their_own_constants() {
    use crate::kernels::nearest::{Floors, Metric};
    use crate::ops::config::Weights;
    let mut planner = super::planner(crate::atoms::sm86());
    let problem = |k| crate::ops::config::NearestProblem {
        n: 1 << 14,
        m: 1 << 14,
        d: 128,
        dtype: ScalarDType::Float16,
        metric: Metric::SquaredL2,
        floors: Floors::default(),
        groups: false,
        k,
    };
    let cost =
        |planner: &Planner, p: &crate::ops::config::NearestProblem, cfg: &crate::kernels::nearest::NearestCfg| {
            p.admit(planner, cfg).expect("admitted").0
        };
    let zero = |cluster: &str| match cluster {
        "pair" => Weights { pair: 0.0, ..Weights::same() },
        "deal" => Weights { deal: 0.0, ..Weights::same() },
        "merge" => Weights { merge: 0.0, ..Weights::same() },
        _ => Weights { scan: 0.0, ..Weights::same() },
    };
    let least = problem(1);
    let cfg = planner.nearest_ranked(least)[0];
    let same = cost(&planner, &least, &cfg);
    for cluster in ["deal", "merge", "scan"] {
        planner.weights = zero(cluster);
        assert_eq!(cost(&planner, &least, &cfg), same, "a least holds no {cluster} work");
    }
    planner.weights = zero("pair");
    assert!(cost(&planner, &least, &cfg) < same, "a least pays its pair epilogue");
    planner.weights = Weights::same();

    let lists = problem(64);
    let ranked = planner.nearest_ranked(lists);
    let (list, matrix) = ranked
        .iter()
        .find(|c| c.select.is_none())
        .zip(ranked.iter().find(|c| c.select.is_some()))
        .expect("a list and a score-matrix config ranked");
    let lists_of = [(list, &["pair", "deal", "merge"][..]), (matrix, &["pair", "scan"][..])];
    for (cfg, clusters) in lists_of {
        let same = cost(&planner, &lists, cfg);
        for cluster in clusters {
            planner.weights = zero(cluster);
            assert!(cost(&planner, &lists, cfg) != same, "a {cluster} config moves with its cluster");
        }
    }
}

/// The op layer's own tuning reads the same line the harness derives: its
/// replayed pick is the line's fastest candidate.
#[test]
fn the_op_layers_tuning_reads_the_same_line() {
    let dir = Scratch::new("rank-op");
    let planner = store(&dir, |at| [900, 800, 700, 600][at]);
    let ranked = planner.gemm_ranked(1, 512, 512, 512, false);
    let spec =
        GemmSpec { m: 512, n: 512, k: 512, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg: ranked[0] };
    let pick =
        tuned::linear(&TuneStore::replay(dir.to_path_buf()), &planner, svod_dtype::DType::BFloat16, &ranked, spec).cfg;
    assert_eq!(pick, representatives(&ranked)[3], "the line's fastest, at 600 ns");
}

/// The claim diff's mappings that hold: a trip's matrix steps in the
/// target's atom instructions and the config's declared shared memory are
/// exact on every family's sample — the model counts what the program does.
#[test]
fn the_claimed_steps_and_smem_hold_their_programs() {
    let diffed = rank::claims(&sm86());
    assert!(diffed.len() >= 8, "{:?}", diffed.iter().map(|d| &d.name).collect::<Vec<_>>());
    for d in &diffed {
        let row = |what: &str| d.rows.iter().find(|(w, ..)| *w == what).copied().expect(what);
        let (_, predicted, program, holds) = row("steps/trip");
        assert!((predicted - program).abs() < 0.5 && holds, "{}: {predicted} steps against {program}", d.name);
        let (_, predicted, program, holds) = row("smem");
        assert_eq!((predicted, program, holds), (predicted, predicted, true), "{}: shared bytes", d.name);
    }
}

/// A GEMM's epilogue claim enters the ranking, and the diff's VALU row
/// predicts a program's instruction mix from the claim itself: the gelu's
/// forty tile operations an element, [128,256,64] at 0.7 % under.
#[test]
fn the_gemm_epilogue_claim_holds_its_program() {
    for d in &rank::claims(&sm86()) {
        if !d.name.starts_with("gemm") {
            continue;
        }
        let (_, predicted, program, holds) =
            d.rows.iter().find(|(w, ..)| *w == "valu ops/trip").copied().expect("the valu row");
        assert!(holds, "{}: {predicted} VALU against {program}", d.name);
    }
}

/// The claim diff's reported opacity: the program moves the model's fill
/// bytes once to about twice over — the ring's staging and the fill
/// layout's per-thread assignment against the model's block mean — and the
/// VALU count only sometimes maps (a GEMM's epilogue the model prices no
/// operations of; a nearest list's dealt rounds the model counts as an
/// average). Bounded, not pinned.
#[test]
fn the_claimed_fill_and_valu_are_bounded_not_pinned() {
    for d in &rank::claims(&sm86()) {
        let (_, predicted, program, _) = d.rows.iter().find(|(w, ..)| *w == "fill bytes/trip").expect("the fill");
        let (predicted, program) = (*predicted, *program);
        assert!(
            predicted <= program && program <= 2.5 * predicted.max(1.0),
            "{}: the program moves {program} of {predicted} fill bytes",
            d.name
        );
    }
}
