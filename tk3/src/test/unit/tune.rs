//! The tune store: memo, on-disk round trip, key stability, the rule that
//! nothing unmeasured is stored, and on a CUDA device a real GEMM tuning.

use std::cell::Cell;
use std::path::PathBuf;

use svod_dtype::ScalarDType;
use test_case::test_case;

use crate::atoms::sm86;
use crate::build::BF16;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::ops::config::Planner;
use crate::tune::{self, TuneKey, TuneStore, fingerprint};

fn key(shape: &[usize], candidates: &[usize]) -> TuneKey {
    TuneKey::new("gemm", &sm86(), ScalarDType::BFloat16, shape, &candidates)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("svod-tk3-tune-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// A measurement that records it ran.
fn timed<'a>(ns: &'a [Option<u64>], ran: &'a Cell<bool>) -> impl FnOnce() -> Vec<Option<u64>> + 'a {
    move || {
        ran.set(true);
        ns.to_vec()
    }
}

/// The fastest is stored; a second store on the directory reads it back
/// without measuring; a kernel change (another program fingerprint) re-measures.
#[test]
fn a_winner_round_trips_through_the_store() {
    let dir = scratch("round-trip");
    let k = key(&[1, 704, 512, 512], &[1, 2, 3]);
    let ran = Cell::new(false);
    let chosen = TuneStore::at(Some(dir.clone())).select(&k, 3, || 7, timed(&[Some(300), Some(100), None], &ran));
    assert_eq!((chosen, ran.get()), (Some(1), true));
    let text = std::fs::read_to_string(dir.join("sm_86_28sm-v0.2.0.txt")).expect("the store file");
    assert_eq!(text, format!("{} 1 100\n", k.line(7)));

    let again = TuneStore::at(Some(dir.clone()));
    let ran = Cell::new(false);
    assert_eq!((again.select(&k, 3, || 7, timed(&[None; 3], &ran)), ran.get()), (Some(1), false));

    let changed = TuneStore::at(Some(dir.clone()));
    let ran = Cell::new(false);
    assert_eq!(changed.select(&k, 3, || 8, timed(&[Some(5), Some(9), Some(9)], &ran)), Some(0));
    assert!(ran.get(), "another kernel fingerprint is a miss");
    assert_eq!(changed.entries(&k).len(), 2, "both lines are kept");
    let _ = std::fs::remove_dir_all(dir);
}

/// A warm memo neither fingerprints nor measures.
#[test]
fn the_memo_answers_without_building() {
    let store = TuneStore::at(None);
    let k = key(&[4096, 4096, 4096], &[0, 1]);
    assert_eq!(store.select(&k, 2, || 1, || vec![Some(2), Some(1)]), Some(1));
    let fingerprinted = Cell::new(false);
    let ran = Cell::new(false);
    let fp = || {
        fingerprinted.set(true);
        1
    };
    assert_eq!(store.select(&k, 2, fp, timed(&[None; 2], &ran)), Some(1));
    assert!(!fingerprinted.get() && !ran.get());
}

/// Nothing measured: no choice, and no line on disk.
#[test]
fn nothing_measured_is_never_stored() {
    let dir = scratch("unmeasured");
    let store = TuneStore::at(Some(dir.clone()));
    let k = key(&[64, 64, 64], &[0, 1]);
    assert_eq!(store.select(&k, 2, || 1, || vec![None, None]), None);
    assert!(store.entries(&k).is_empty());
    assert_eq!(store.pick(&k, &[10, 20], |_| unreachable!("memoized")), 10, "the first candidate");
    let _ = std::fs::remove_dir_all(dir);
}

/// A stored index past the candidate count is a miss.
#[test]
fn a_stale_index_is_a_miss() {
    let dir = scratch("stale");
    let k = key(&[8, 8, 8], &[0, 1, 2]);
    TuneStore::at(Some(dir.clone())).select(&k, 3, || 1, || vec![None, None, Some(1)]);
    let ran = Cell::new(false);
    assert_eq!(TuneStore::at(Some(dir.clone())).select(&k, 2, || 1, timed(&[Some(1), None], &ran)), Some(0));
    assert!(ran.get());
    let _ = std::fs::remove_dir_all(dir);
}

#[test_case(&[1, 704, 512, 512], &[0, 1], true; "the same inputs")]
#[test_case(&[1, 704, 512, 2048], &[0, 1], false; "another shape")]
#[test_case(&[1, 704, 512, 512], &[1, 0], false; "another candidate order")]
fn keys_are_stable(shape: &[usize], candidates: &[usize], same: bool) {
    let base = key(&[1, 704, 512, 512], &[0, 1]);
    let other = key(shape, candidates);
    assert_eq!(base.line(3) == other.line(3), same);
    assert!(base.line(3).starts_with("gemm|sm_86-28sm|BFloat16|1x704x512x512|"), "{}", base.line(3));
    assert_ne!(base.line(3), base.line(4), "the program fingerprint is part of the line");
}

/// `SVOD_TK3_TUNE=0`'s override: off and back on.
#[test]
fn tuning_can_be_turned_off() {
    tune::set_enabled(false);
    assert!(!tune::enabled());
    tune::set_enabled(true);
    assert!(tune::enabled());
    tune::set_enabled(false);
}

/// Tunes a Nemotron GEMM on the device into a scratch directory; a second
/// store on that directory answers from disk without measuring.
#[test]
fn a_gemm_tunes_on_the_device() {
    let spec = svod_dtype::default_device::default_device();
    let Some(target) =
        matches!(spec, svod_dtype::DeviceSpec::Cuda { .. }).then(|| crate::atoms::Target::for_device(&spec)).flatten()
    else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let (m, n, k) = (704, 512, 512);
    let cfgs = Planner::new(target.clone()).gemm_candidates(1, m, n, k, false);
    let batch = Batch::Var { name: "b".into(), min: 1, max: 1 };
    let build = |cfg: GemmCfg| {
        let spec = GemmSpec { m, n, k, batch: batch.clone(), epilogue: Epilogue::default(), cfg };
        vec![(gemm::<BF16>(&spec), cfg.lowering(target.clone()))]
    };
    let key = TuneKey::new("gemm", &target, ScalarDType::BFloat16, &[1, m, n, k], &cfgs);
    let dir = scratch("device");
    let start = std::time::Instant::now();
    let chosen = TuneStore::at(Some(dir.clone())).pick(&key, &cfgs, build);
    eprintln!("tuned {} candidates in {:.1} s: {chosen:?}", cfgs.len(), start.elapsed().as_secs_f64());
    let programs = || fingerprint(&cfgs.iter().map(|&c| build(c)).collect::<Vec<_>>());
    let hit = TuneStore::at(Some(dir.clone())).select(&key, cfgs.len(), programs, || panic!("measured twice"));
    assert_eq!(hit.map(|i| cfgs[i]), Some(chosen));
    let _ = std::fs::remove_dir_all(dir);
}
