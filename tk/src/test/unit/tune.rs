//! Tests for the first-use tuning store ([`crate::tune`]): the memo, the
//! on-disk round trip, the rule that nothing unmeasured is cached, and — on a
//! GPU — a real first-use measurement of the GEMM table.

use std::path::PathBuf;

use svod_dtype::{DType, DeviceSpec, GpuArch};

use crate::tune::{TuneKey, TuneStore};

fn key(kernel: &'static str, shape: &[usize], builds: &[u128]) -> TuneKey {
    let arch = GpuArch::Amd(svod_dtype::AmdArch::Gfx1151);
    TuneKey::new(kernel, &DeviceSpec::Cpu, arch, shape, builds)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("svod-tk-tune-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// The first selection measures and keeps the fastest; a second store on the
/// same directory reads it back without measuring; a different candidate list
/// (a table or kernel change) is a different key.
#[test]
fn a_measured_winner_round_trips_through_the_store() {
    let dir = scratch("round-trip");
    let store = TuneStore::at(Some(dir.clone()));
    let k = key("gemm_nt", &[1024, 1024, 6144], &[10, 20, 30]);
    let mut measured = false;
    let chosen = store.select_with(&k, 3, || {
        measured = true;
        vec![Some(300), Some(100), Some(200)]
    });
    assert_eq!((chosen, measured), (Some(1), true));

    let again = TuneStore::at(Some(dir.clone()));
    let mut ran = false;
    let k2 = key("gemm_nt", &[1024, 1024, 6144], &[10, 20, 30]);
    assert_eq!(
        again.select_with(&k2, 3, || {
            ran = true;
            vec![None; 3]
        }),
        Some(1)
    );
    assert!(!ran, "a stored winner is not re-measured");

    assert_ne!(key("gemm_nt", &[1024, 1024, 6144], &[10, 20, 31]), k2, "the candidate kernels are part of the key");
    let _ = std::fs::remove_dir_all(dir);
}

/// A candidate that cannot run is skipped, and when none can, nothing is kept:
/// the caller falls back to its static choice and the next call measures again.
#[test]
fn unmeasured_candidates_are_never_cached() {
    let dir = scratch("unmeasured");
    let store = TuneStore::at(Some(dir.clone()));
    let k = key("fa", &[8, 512, 16, 128], &[1, 2]);
    assert_eq!(store.select_with(&k, 2, || vec![None, None]), None);
    assert_eq!(store.select_with(&k, 2, || vec![None, Some(5)]), Some(1), "the runnable candidate wins");
    let _ = std::fs::remove_dir_all(dir);
}

/// A stored index past the current candidate count (a hand-edited file) is
/// ignored, and a store with no directory still memoizes within the process.
#[test]
fn a_stale_index_is_ignored_and_a_memory_store_memoizes() {
    let dir = scratch("stale");
    let store = TuneStore::at(Some(dir.clone()));
    let k = key("gemm_nt", &[64, 64, 192], &[7, 8, 9]);
    assert_eq!(store.select_with(&k, 3, || vec![Some(10), Some(9), Some(8)]), Some(2));
    let fresh = TuneStore::at(Some(dir.clone()));
    let mut measured = false;
    assert_eq!(
        fresh.select_with(&k, 2, || {
            measured = true;
            vec![Some(1), Some(2)]
        }),
        Some(0)
    );
    assert!(measured, "a stored index past the candidate count is measured again");

    let memory = TuneStore::at(None);
    let k = key("norm", &[4096, 1024], &[1, 2]);
    let mut runs = 0;
    assert_eq!(
        memory.select_with(&k, 2, || {
            runs += 1;
            vec![Some(2), Some(1)]
        }),
        Some(1)
    );
    assert_eq!(
        memory.select_with(&k, 2, || {
            runs += 1;
            vec![Some(0), Some(0)]
        }),
        Some(1)
    );
    assert_eq!(runs, 1, "measured once, then memoized");
    let _ = std::fs::remove_dir_all(dir);
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
