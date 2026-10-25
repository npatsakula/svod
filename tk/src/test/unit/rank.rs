//! The rank harness's own math and replay: Spearman over ties, and a
//! synthetic store replayed through the very keys the op layer writes —
//! the digest assert holding, a drifted list refused and counted.

use test_case::test_case;

use super::Scratch;
use svod_dtype::ScalarDType;

use crate::atoms::sm86;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec};
use crate::ops::config::{GemmProblem, Planner, representatives};
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
    let order = Scratch::new("rank-order");
    let p = predicted(&store(&order, |at| at as u64));
    assert!(p.windows(2).all(|w| w[0] <= w[1]) && p[0] < p[3], "the model's order varies: {p:?}");
    let agreed = (ns(1.0) as i64 - ns(0.0) as i64).signum() > 0;
    let dir = Scratch::new(if agreed { "rank-agree" } else { "rank-invert" });
    store(&dir, |at| ns(p[at]));
    let devices = rank::replay(&dir, None, Some(&["gemm"]));
    assert_eq!(devices.len(), 1, "the one device file");
    let family = &devices[0].families[0];
    assert_eq!((family.name, family.lines.len(), family.skipped), ("gemm", 1, 0));
    let rho = family.lines[0].rho(&devices[0].planner).expect("the costs give the order variance");
    assert!(if agreed { rho > 0.99 } else { rho < -0.99 }, "{rho}");
    assert_eq!(family.lines[0].top1(&devices[0].planner), agreed);
}

/// A choice line whose candidates digest names a list no ranking of the
/// shape makes is skipped and counted, never read as one.
#[test]
fn a_choice_line_no_ranking_makes_is_skipped() {
    let dir = Scratch::new("rank-drift");
    store(&dir, |at| 100 + at as u64);
    let file = TuneStore::at(Some(dir.to_path_buf())).file("sm_86-28sm").expect("a file");
    let text = std::fs::read_to_string(&file).expect("read");
    std::fs::write(
        &file,
        format!("{text}choice|linear|sm_86-28sm|BFloat16|1x512x512x512|deadbeefdeadbeef|0|0 0 100 1 200 2 300\n"),
    )
    .expect("written");
    let devices = rank::replay(&dir, None, Some(&["gemm"]));
    let family = &devices[0].families[0];
    assert_eq!((family.lines.len(), family.skipped), (1, 1), "the made line replays, the dead one counts");
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
