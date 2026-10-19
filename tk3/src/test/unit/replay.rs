//! A tune store read back without a device ([`TuneStore::replay`]): which
//! of a key's lines it reads whatever compiler and kernel build wrote them,
//! what it reports per key, that it writes nothing, and that an op's tuning
//! replayed through it reads the keys the op layer writes and follows its
//! admission, choice and pipeline stages to the config they name.

use std::path::{Path, PathBuf};

use svod_dtype::{DType, ScalarDType};
use test_case::test_case;

use crate::atoms::sm86;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec};
use crate::ops::config::{Tiled, representatives};
use crate::ops::{DEVICE_TEST, tuned};
use crate::tune::{self, Entry, FORMAT, Found, Kind, TuneKey, TuneStore};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("svod-tk3-replay-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Writes `lines` (`line numbers`) as `key`'s device file under `dir`.
fn write(dir: &Path, key: &TuneKey, lines: &[String]) -> PathBuf {
    let file = TuneStore::at(Some(dir.to_path_buf())).file(&key.device).expect("a file");
    std::fs::write(&file, format!("format {FORMAT}\n{}\n", lines.join("\n"))).expect("written");
    file
}

/// Every file under `dir`, with its contents.
fn tree(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = vec![];
    for entry in std::fs::read_dir(dir).expect("a dir").flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(tree(&path));
        } else {
            out.push((path.clone(), std::fs::read(&path).expect("readable")));
        }
    }
    out.sort();
    out
}

fn key() -> TuneKey {
    TuneKey::new("linear", &sm86(), 0, ScalarDType::BFloat16, &[1, 64, 64, 64], &"a list")
}

/// The line a replay reads of a key's: the one of this build's programs,
/// else other builds' lines, all alike or the first of them; none of the
/// key's, with the lines of its shape under other candidates counted. The
/// compiler is never matched.
#[test_case(&[(7, 3, 11)], 3 => (Found::Own, Some(11)); "this build's line under another compiler")]
#[test_case(&[(7, 4, 11), (8, 3, 12)], 3 => (Found::Own, Some(12)); "this build's line among others")]
#[test_case(&[(7, 4, 11), (8, 5, 11)], 3 => (Found::Built { lines: 2 }, Some(11)); "other builds agreeing")]
#[test_case(&[(7, 4, 13), (7, 5, 11)], 3 => (Found::Disagree { lines: 2 }, Some(13)); "other builds disagreeing")]
#[test_case(&[], 3 => (Found::Missing { others: 1 }, None); "no line of the key")]
fn a_replay_reads_a_keys_line_whatever_compiler_wrote_it(
    lines: &[(u64, u64, u64)],
    programs: u64,
) -> (Found, Option<u64>) {
    let name: String = format!("{lines:?}").chars().filter(char::is_ascii_digit).collect();
    let dir = scratch(&format!("lines-{name}"));
    let key = key();
    let other = TuneKey { candidates: key.candidates ^ 1, ..key.clone() };
    let mut text: Vec<String> = lines
        .iter()
        .map(|&(compiler, programs, v)| {
            format!("{} {v}", TuneKey { compiler, ..key.clone() }.line(Kind::Value, programs))
        })
        .collect();
    text.push(format!("{} 99", other.line(Kind::Value, programs)));
    write(&dir, &key, &text);
    let store = TuneStore::replay(dir.clone());
    let v = store.value(&key, || programs, || panic!("a replay measures nothing"));
    let lookups = store.lookups();
    assert_eq!(lookups.len(), 1);
    assert_eq!(lookups[0].key, key.candidates(Kind::Value));
    assert_eq!(lookups[0].entry, v.map(Entry::Value));
    let _ = std::fs::remove_dir_all(dir);
    (lookups[0].found, v)
}

/// A replay changes nothing under its root: no line is stored for what it
/// cannot find, no memo of what programs emit is kept or pruned, and the
/// costs read from the same directory open nothing for writing either.
#[test]
fn a_replay_writes_nothing() {
    let dir = scratch("writes");
    let planner = super::planner(sm86());
    let ranked = planner.gemm_ranked(1, 256, 256, 256, false);
    let file = write(&dir, &key(), &[format!("{} 5", key().line(Kind::Value, 1))]);
    std::fs::create_dir_all(dir.join("emitted")).expect("emitted dir");
    let stale = dir.join("emitted/0000000000000000.txt");
    std::fs::write(&stale, "1 2\n").expect("a memo");
    let old = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1);
    std::fs::File::options().write(true).open(&stale).and_then(|f| f.set_modified(old)).expect("aged");
    let before = tree(&dir);
    let spec =
        GemmSpec { m: 256, n: 256, k: 256, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg: ranked[0] };
    let store = TuneStore::replay(dir.clone());
    tuned::linear(&store, &planner, DType::BFloat16, &ranked, spec);
    assert!(store.lookups().iter().all(|l| matches!(l.found, Found::Missing { .. })), "{:?}", store.lookups());
    // No probe line: every probe is unmeasured, each fingerprinted to look.
    assert!(crate::probe::stored(&dir, &sm86()).is_some_and(|m| !m.unmeasured.is_empty()));
    assert_eq!(tree(&dir), before, "the replay changed its root");
    assert!(file.exists());
    let _ = std::fs::remove_dir_all(dir);
}

/// What `linear` and the GEMM probes key their first store line by: the
/// op's name (`linear`) and its salt, the batch and epilogue, or the
/// kernel's name and [`DEVICE_TEST`]; then the representatives of the
/// ranking, as Debug text. A change here orphans every stored choice.
#[test_case(false; "linear")]
#[test_case(true; "the GEMM probes")]
fn a_gemm_tuning_reads_the_key_the_op_layer_writes(probe: bool) {
    let planner = super::planner(sm86());
    let ranked = planner.gemm_ranked(1, 512, 512, 512, false);
    let spec =
        GemmSpec { m: 512, n: 512, k: 512, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg: ranked[0] };
    let store = TuneStore::replay(scratch(&format!("key-{probe}")));
    let reps = representatives(&ranked);
    let (op, candidates) = if probe {
        tuned::gemm_probe(&store, &planner, DType::BFloat16, &ranked, spec);
        ("gemm", format!("{DEVICE_TEST:?} {reps:?}"))
    } else {
        tuned::linear(&store, &planner, DType::BFloat16, &ranked, spec);
        ("linear", format!("{:?} {reps:?}", (Batch::Static(1), Epilogue::DEFAULT)))
    };
    let want = TuneKey::new(op, &sm86(), 0, ScalarDType::BFloat16, &[1, 512, 512, 512], &candidates);
    let lookups = store.lookups();
    assert_eq!(lookups[0].kind, Kind::Admitted);
    assert_eq!(lookups[0].key, want.candidates(Kind::Admitted));
}

/// A replayed tuning follows the stored stages: the admitted indices name
/// configs of the ranking's representatives, the choice orders those
/// candidates, and with the leader alone within the contender band (its
/// twins' stage unstored) the leader is the pick.
#[test]
fn a_replayed_gemm_follows_the_admission_and_the_choice() {
    let dir = scratch("chain");
    let planner = super::planner(sm86());
    let ranked = planner.gemm_ranked(1, 512, 512, 512, false);
    let reps = representatives(&ranked);
    let salt = format!("{:?}", (Batch::Static(1), Epilogue::DEFAULT));
    let at = |list: &[GemmCfg]| {
        TuneKey::new("linear", &sm86(), 0, ScalarDType::BFloat16, &[1, 512, 512, 512], &format!("{salt} {list:?}"))
    };
    let (admitted, candidates) = (at(&reps), at(&[reps[2], reps[1]]));
    write(
        &dir,
        &admitted,
        &[
            format!("{} 2 1", admitted.line(Kind::Admitted, 1)),
            format!("{} 1 900 0 2000", candidates.line(Kind::Choice, 1)),
        ],
    );
    let spec =
        GemmSpec { m: 512, n: 512, k: 512, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg: ranked[0] };
    let store = TuneStore::replay(dir.clone());
    let pick = tuned::linear(&store, &planner, DType::BFloat16, &ranked, spec).cfg;
    assert_eq!(pick, reps[1], "the choice's fastest, the second admitted");
    let lookups = store.lookups();
    let found: Vec<(Kind, Found)> = lookups.iter().map(|l| (l.kind, l.found)).collect();
    assert_eq!(found[..2], [(Kind::Admitted, Found::Built { lines: 1 }), (Kind::Choice, Found::Built { lines: 1 })]);
    // The twins' stage runs only where the leader has twins to time.
    let twins = reps[1].twins(&ranked).len();
    assert_eq!(lookups.len(), if twins >= 2 { 3 } else { 2 }, "{lookups:?}");
    assert!(tune::enabled());
    let _ = std::fs::remove_dir_all(dir);
}
