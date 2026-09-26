//! Tests for the first-use tuning store ([`crate::tune`]): the memo, the
//! on-disk round trip, the rule that nothing unmeasured is cached, and — on a
//! GPU — a real first-use measurement of the GEMM table.

use std::path::PathBuf;

use svod_dtype::{DType, DeviceSpec, GpuArch};
use test_case::test_case;

use crate::tune::{TuneKey, TuneStore};

/// A key over `candidates` (the cheap identity of the candidate set the memo is
/// keyed by) — `builds`, the candidate graphs' fingerprints, reaches the store
/// separately, so it is supplied per call.
fn key(kernel: &'static str, shape: &[usize], candidates: &[usize]) -> TuneKey {
    let arch = GpuArch::Amd(svod_dtype::AmdArch::Gfx1151);
    TuneKey::new(kernel, &DeviceSpec::Cpu, arch, shape, &candidates)
}

/// A fingerprint closure that records whether it ran: the memo must answer
/// without it, since building the candidate graphs is what it costs.
fn builds(digests: &[u128], ran: &std::cell::Cell<bool>) -> impl FnOnce() -> Vec<u128> {
    move || {
        ran.set(true);
        digests.to_vec()
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("svod-tk-tune-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// The first selection measures and keeps the fastest; a second store on the
/// same directory reads it back without measuring; a different candidate set is
/// a different key, and so is the same set built into different kernels.
#[test]
fn a_measured_winner_round_trips_through_the_store() {
    let dir = scratch("round-trip");
    let store = TuneStore::at(Some(dir.clone()));
    let k = key("gemm_nt", &[1024, 1024, 6144], &[1, 2, 3]);
    let (mut measured, fingerprinted) = (false, std::cell::Cell::new(false));
    let chosen = store.select_with(&k, 3, builds(&[10, 20, 30], &fingerprinted), || {
        measured = true;
        vec![Some(300), Some(100), Some(200)]
    });
    assert_eq!((chosen, measured, fingerprinted.get()), (Some(1), true, true));

    let again = TuneStore::at(Some(dir.clone()));
    let mut ran = false;
    let k2 = key("gemm_nt", &[1024, 1024, 6144], &[1, 2, 3]);
    assert_eq!(
        again.select_with(&k2, 3, builds(&[10, 20, 30], &std::cell::Cell::new(false)), || {
            ran = true;
            vec![None; 3]
        }),
        Some(1)
    );
    assert!(!ran, "a stored winner is not re-measured");

    // The same candidate set built into different kernels shares the memo key
    // but not the store line, so the store re-measures it.
    let kernel_change = TuneStore::at(Some(dir.clone()));
    let mut remeasured = false;
    assert_eq!(
        kernel_change.select_with(&k2, 3, builds(&[10, 20, 31], &std::cell::Cell::new(false)), || {
            remeasured = true;
            vec![Some(1), Some(2), Some(3)]
        }),
        Some(0)
    );
    assert!(remeasured, "a kernel change re-measures");

    assert_ne!(key("gemm_nt", &[1024, 1024, 6144], &[1, 2, 4]), k2, "the candidate set is part of the key");
    let _ = std::fs::remove_dir_all(dir);
}

/// A memo hit answers without building a single candidate graph: the whole point
/// of keying the memo by the request rather than by the built kernels, since a
/// plan asks the same shape once per node.
#[test]
fn a_memoized_choice_builds_no_candidate() {
    let store = TuneStore::at(None);
    let k = key("gemm_nt", &[4096, 1024, 6144], &[1, 2, 3]);
    let first = std::cell::Cell::new(false);
    assert_eq!(store.select_with(&k, 3, builds(&[7, 8, 9], &first), || vec![Some(3), Some(1), Some(2)]), Some(1));
    assert!(first.get(), "the first call fingerprints the candidates for the store line");

    let again = std::cell::Cell::new(false);
    assert_eq!(store.select_with(&k, 3, builds(&[7, 8, 9], &again), || panic!("a memo hit must not measure")), Some(1));
    assert!(!again.get(), "a memo hit must not build the candidate kernels");
}

/// A candidate that cannot run is skipped, and when none can, nothing is kept:
/// the caller falls back to its static choice and the next call measures again.
#[test]
fn unmeasured_candidates_are_never_cached() {
    let dir = scratch("unmeasured");
    let store = TuneStore::at(Some(dir.clone()));
    let k = key("fa", &[8, 512, 16, 128], &[1, 2]);
    let seen = std::cell::Cell::new(false);
    assert_eq!(store.select_with(&k, 2, builds(&[1, 2], &seen), || vec![None, None]), None);
    assert_eq!(
        store.select_with(&k, 2, builds(&[1, 2], &seen), || vec![None, Some(5)]),
        Some(1),
        "the runnable candidate wins"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A stored index past the current candidate count (a hand-edited file) is
/// ignored, and a store with no directory still memoizes within the process.
#[test]
fn a_stale_index_is_ignored_and_a_memory_store_memoizes() {
    let dir = scratch("stale");
    let store = TuneStore::at(Some(dir.clone()));
    let k = key("gemm_nt", &[64, 64, 192], &[7, 8, 9]);
    let seen = std::cell::Cell::new(false);
    assert_eq!(store.select_with(&k, 3, builds(&[7, 8, 9], &seen), || vec![Some(10), Some(9), Some(8)]), Some(2));
    let fresh = TuneStore::at(Some(dir.clone()));
    let mut measured = false;
    assert_eq!(
        fresh.select_with(&k, 2, builds(&[7, 8, 9], &seen), || {
            measured = true;
            vec![Some(1), Some(2)]
        }),
        Some(0)
    );
    assert!(measured, "a stored index past the candidate count is measured again");

    let memory = TuneStore::at(None);
    let k = key("norm", &[4096, 1024], &[1, 2]);
    let seen = std::cell::Cell::new(false);
    let mut runs = 0;
    assert_eq!(
        memory.select_with(&k, 2, builds(&[1, 2], &seen), || {
            runs += 1;
            vec![Some(2), Some(1)]
        }),
        Some(1)
    );
    assert_eq!(
        memory.select_with(&k, 2, builds(&[1, 2], &seen), || {
            runs += 1;
            vec![Some(0), Some(0)]
        }),
        Some(1)
    );
    assert_eq!(runs, 1, "measured once, then memoized");
    let _ = std::fs::remove_dir_all(dir);
}

/// `set_enabled` reaches the launches of its own thread only. The test runner
/// gives each test a thread, so a kernel test that turns tuning off leaves the
/// model test running beside it tuning as a user's process would.
#[test]
fn set_enabled_reaches_only_its_own_thread() {
    let environment = std::env::var("SVOD_TK_TUNE").map(|v| v != "0").unwrap_or(true);
    crate::tune::set_enabled(false);
    assert!(!crate::tune::enabled(), "the setting holds on its own thread");
    let (before, after) = std::thread::spawn(|| {
        let before = crate::tune::enabled();
        crate::tune::set_enabled(true);
        (before, crate::tune::enabled())
    })
    .join()
    .expect("the other thread");
    assert_eq!(before, environment, "another thread follows the environment, not this thread's setting");
    assert!(after, "and holds its own setting");
    assert!(!crate::tune::enabled(), "whose setting does not come back here");
}

/// On a supported GPU, a first request measures the GEMM table for a shape and
/// records one line; the winner is a table tile that tiles the shape.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib tune::gemm_first_use -- --ignored`.
#[test]
#[ignore]
fn gemm_first_use_measures_the_table_once_gpu() {
    use crate::kernels::gemm::{Epilogue, GEMM_NT_SUPPORTED_ARCHS, GemmPolicy};

    if !super::device_supported(GEMM_NT_SUPPORTED_ARCHS) {
        eprintln!("skip gemm_first_use_measures_the_table_once_gpu: no supported device / toolchain");
        return;
    }
    let dir = scratch("gemm-gpu");
    let store = TuneStore::at(Some(dir.clone()));
    let spec = svod_tensor::Tensor::empty(&[1], DType::Float32).device();
    let arch = crate::target::resolve_arch(&spec).expect("a GPU arch");
    let policy = GemmPolicy::for_device(&spec, arch);
    let (m, k, n) = (256usize, 1024usize, 1024usize);
    let cfg = policy.tuned(&store, &spec, arch, &DType::BFloat16, (m, k, n), Epilogue::Plain).expect("a tile");
    assert!(policy.tiles.contains(&cfg) && cfg.tiles(m, k, n), "the winner is a table tile that tiles the shape");
    let files: Vec<_> = std::fs::read_dir(&dir).expect("store dir").flatten().collect();
    assert_eq!(files.len(), 1, "one file per device");
    let text = std::fs::read_to_string(files[0].path()).expect("store file");
    assert_eq!(text.lines().count(), 1, "one line per shape: {text}");
    assert!(text.starts_with("gemm_nt|"), "{text}");
    assert_eq!(policy.tuned(&store, &spec, arch, &DType::BFloat16, (m, k, n), Epilogue::Plain), Some(cfg));
    let _ = std::fs::remove_dir_all(dir);
}

/// The attention split's first use measures the policy's candidates (each a
/// partial + merge pair) once and records one line; the winner is a candidate.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib tune::sq_attention_first_use -- --ignored`.
#[test_case(1, 20, true; "one shared cache")]
#[test_case(5, 640, true; "whisper large's packed cross cache")]
#[test_case(5, 20, false; "a cache per row")]
#[ignore]
fn sq_attention_first_use_measures_the_splits_once_gpu(kv_batch: usize, h_total: usize, cache_map: bool) {
    use crate::kernels::sq_attention::{HeadSelection, SQ_ATTENTION_SUPPORTED_ARCHS, SqGeom, SqPolicy};

    if !super::device_supported(SQ_ATTENTION_SUPPORTED_ARCHS) {
        eprintln!("skip sq_attention_first_use_measures_the_splits_once_gpu: no supported device / toolchain");
        return;
    }
    let spec = svod_tensor::Tensor::empty(&[1], DType::Float32).device();
    let arch = crate::target::resolve_arch(&spec).expect("a GPU arch");
    let policy = SqPolicy::for_device(&spec, arch);
    let (b, n, h, d) = (5, 1500, 20, 64);
    let candidates = policy.candidates(b, h, n);
    if candidates.len() < 2 {
        eprintln!("skip sq_attention_first_use_measures_the_splits_once_gpu: the family keeps one split");
        return;
    }
    let dir = scratch(&format!("sq-gpu-{kv_batch}-{h_total}-{cache_map}"));
    let store = TuneStore::at(Some(dir.clone()));
    let heads = HeadSelection { count: h, total: h_total, offset: h_total - h };
    let geom = SqGeom { b, kv_batch, n, heads, d, kv: DType::Float16 };
    let split = policy.tuned(&store, &spec, arch, &geom, cache_map);
    assert!(candidates.contains(&split), "the winner {split} is a candidate of {candidates:?}");
    let files: Vec<_> = std::fs::read_dir(&dir).expect("store dir").flatten().collect();
    assert_eq!(files.len(), 1, "one file per device");
    let text = std::fs::read_to_string(files[0].path()).expect("store file");
    assert_eq!(text.lines().count(), 1, "one line per shape: {text}");
    assert!(text.starts_with("sq_attention|"), "{text}");
    assert_eq!(policy.tuned(&store, &spec, arch, &geom, cache_map), split);
    let _ = std::fs::remove_dir_all(dir);
}
