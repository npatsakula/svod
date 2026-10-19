//! The tune store: memo, on-disk round trip of choices and values, key
//! stability, the rule that nothing unmeasured is stored, a key measured
//! once under concurrent asks, a store hit compiling nothing, a run's
//! stamps and the lost-stamp guard, what counts as a spill, the admission
//! memo, and on a CUDA device a real GEMM tuning.

use std::cell::Cell;
use std::path::PathBuf;
use std::time::Duration;

use svod_dtype::ScalarDType;
use svod_runtime::KernelResources;
use test_case::test_case;

use crate::atoms::{Device, sm86};
use crate::build::BF16;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::tune::{self, Bench, FORMAT, Kind, ScratchPool, TuneKey, TuneStore, fingerprint};

/// A host key: no device, so no compiler digest.
fn key(shape: &[usize], candidates: &[usize]) -> TuneKey {
    TuneKey::new("gemm", &sm86(), 0, ScalarDType::BFloat16, shape, &candidates)
}

/// The store file of [`key`]'s device under `dir`.
fn store_file(dir: &std::path::Path) -> String {
    let path = dir.join(format!("sm_86_28sm-v{}.txt", env!("CARGO_PKG_VERSION")));
    std::fs::read_to_string(path).expect("the store file")
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
    assert_eq!(store_file(&dir), format!("format {FORMAT}\n{} 1 100\n", k.line(Kind::Choice, 7)));

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
    let _ = std::fs::remove_dir_all(dir);
}

/// Two threads asking for one key at once measure it once: the second
/// waits for the first's value.
#[test]
fn a_key_in_flight_is_measured_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let store = TuneStore::at(None);
    let k = key(&[2, 2, 2], &[0, 1]);
    let (measured, start) = (AtomicUsize::new(0), std::sync::Barrier::new(2));
    let ask = || {
        start.wait();
        store.select(
            &k,
            2,
            || 1,
            || {
                measured.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(50));
                vec![Some(2), Some(1)]
            },
        )
    };
    let chosen = std::thread::scope(|s| {
        let other = s.spawn(ask);
        [ask(), other.join().expect("the other asker")]
    });
    assert_eq!((chosen, measured.load(Ordering::SeqCst)), ([Some(1), Some(1)], 1));
}

/// A plain bf16 GEMM's programs at `tile` on sm_86.
fn gemm_programs(tile: [usize; 3]) -> tune::Candidate {
    let cfg = GemmCfg { tile, stages: 2, warps: [2, 2], group_m: 8, unroll: true };
    let spec = GemmSpec { m: 128, n: 128, k: 128, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg };
    vec![(gemm::<BF16>(&spec), cfg.lowering(&Device { target: sm86(), costs: super::costs() }))]
}

/// A choice keys by the candidates' programs, each built once: a store
/// miss compiles each candidate once (and times them where the host's
/// device runs sm_86 programs), and a store hit in another process builds
/// the programs to fingerprint them and compiles nothing.
#[test]
fn a_store_hit_compiles_nothing() {
    let dir = scratch("hit");
    let k = key(&[128, 128, 128], &[0, 1]);
    let tiles = [[64, 64, 32], [128, 64, 32]];
    let builds = Cell::new(0);
    let build = |i: usize| {
        builds.set(builds.get() + 1);
        gemm_programs(tiles[i])
    };
    let programs = fingerprint(&[gemm_programs(tiles[0]), gemm_programs(tiles[1])]);
    let mut bench = Bench::default();
    let missed = TuneStore::at(Some(dir.clone())).choose(&k, &[0, 1], build, &mut bench);
    assert_eq!((builds.get(), bench.built.len()), (2, 2), "a miss builds and compiles each once");
    // Without a device that timed them, what it would have measured, stored
    // as a timing would; with one, its choice is already stored.
    let stored = TuneStore::at(Some(dir.clone())).select(&k, 2, || programs, || vec![Some(9), Some(4)]);
    assert_eq!(stored, missed.or(Some(1)), "the device's choice, else the stored timing's");
    builds.set(0);
    let mut bench = Bench::default();
    let hit = TuneStore::at(Some(dir.clone())).choose(&k, &[0, 1], build, &mut bench);
    assert_eq!((hit, builds.get(), bench.built.len()), (stored, 2, 0));
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

/// A value is stored as a `value` line holding the number; a second store
/// on the directory reads it back without measuring.
#[test]
fn a_value_round_trips_through_the_store() {
    let dir = scratch("value");
    let k = key(&[], &[9]);
    assert_eq!(TuneStore::at(Some(dir.clone())).value(&k, || 7, || Some(1234)), Some(1234));
    assert_eq!(store_file(&dir), format!("format {FORMAT}\n{} 1234\n", k.line(Kind::Value, 7)));
    let measured = || panic!("measured twice");
    assert_eq!(TuneStore::at(Some(dir.clone())).value(&k, || 7, measured), Some(1234));
    let _ = std::fs::remove_dir_all(dir);
}

/// A warm memo answers a value without fingerprinting or measuring; an
/// unmeasured value is memoized as missing and never stored.
#[test_case(Some(5); "measured")]
#[test_case(None; "unmeasured")]
fn the_memo_answers_a_value(measured: Option<u64>) {
    let dir = scratch(&format!("value-memo-{measured:?}"));
    let store = TuneStore::at(Some(dir.clone()));
    let k = key(&[1], &[0]);
    assert_eq!(store.value(&k, || 1, || measured), measured);
    assert_eq!(store.entries(&k).len(), usize::from(measured.is_some()));
    assert_eq!(store.value(&k, || unreachable!("fingerprinted"), || unreachable!("measured")), measured);
    let _ = std::fs::remove_dir_all(dir);
}

/// Choices and values of one device share its file; writing one keeps the other.
#[test]
fn choices_and_values_share_a_file() {
    let dir = scratch("shared");
    let (choice, value) = (key(&[64, 64, 64], &[0, 1]), TuneKey { op: "probe:step", ..key(&[], &[2]) });
    TuneStore::at(Some(dir.clone())).select(&choice, 2, || 1, || vec![Some(9), Some(4)]);
    TuneStore::at(Some(dir.clone())).value(&value, || 2, || Some(42));
    let again = TuneStore::at(Some(dir.clone()));
    assert_eq!(again.entries(&choice).len(), 2);
    assert_eq!(again.select(&choice, 2, || 1, || panic!("measured twice")), Some(1));
    assert_eq!(again.value(&value, || 2, || panic!("measured twice")), Some(42));
    let _ = std::fs::remove_dir_all(dir);
}

/// Another compiler (toolchain, flags) is another line, so it re-measures.
#[test]
fn the_compiler_is_part_of_the_line() {
    let base = key(&[8, 8, 8], &[0, 1]);
    let other = TuneKey { compiler: base.compiler ^ 1, ..base.clone() };
    assert_ne!(base.line(Kind::Choice, 3), other.line(Kind::Choice, 3));
    let line = base.line(Kind::Choice, 3);
    assert!(line.contains(&format!("|{:016x}|", base.compiler)), "{line}");
}

/// A line the store cannot read (an older format, a stray edit) is a miss
/// and is dropped at the next write; a line of another kind under the same
/// key is kept, each kind leading its own line.
#[test]
fn an_unreadable_line_is_dropped_at_the_next_write() {
    let dir = scratch("unreadable");
    let k = key(&[3, 3, 3], &[0, 1]);
    let path = dir.join(format!("sm_86_28sm-v{}.txt", env!("CARGO_PKG_VERSION")));
    let kept = format!("{} 42", k.line(Kind::Value, 1));
    std::fs::write(&path, format!("gemm|sm_86-28sm|BFloat16|3x3x3|0|0|0 1 100\n{kept}\nnonsense\n")).unwrap();
    let store = TuneStore::at(Some(dir.clone()));
    assert_eq!(store.entries(&k).len(), 1);
    assert_eq!(store.select(&k, 2, || 1, || vec![Some(9), Some(4)]), Some(1));
    assert_eq!(store_file(&dir), format!("format {FORMAT}\n{} 1 4\n{kept}\n", k.line(Kind::Choice, 1)));
    let _ = std::fs::remove_dir_all(dir);
}

/// A run's time sums its kernels (a candidate is a kernel and its merge);
/// a kernel without both stamps voids the run rather than shortening it.
#[test_case(&[(Some(10), Some(15)), (Some(20), Some(23))] => Some(8); "kernels summed")]
#[test_case(&[(Some(10), Some(15)), (Some(20), None)] => None; "a lost end")]
#[test_case(&[(None, Some(15)), (Some(20), Some(23))] => None; "a lost start")]
#[test_case(&[(Some(15), Some(10))] => None; "an end before its start")]
#[test_case(&[] => None; "no kernels")]
fn a_run_needs_every_stamp(stamps: &[(Option<u64>, Option<u64>)]) -> Option<u64> {
    tune::stamped(stamps.iter().copied()).map(|d| d.as_nanos() as u64)
}

#[test_case(&[1, 704, 512, 512], &[0, 1], true; "the same inputs")]
#[test_case(&[1, 704, 512, 2048], &[0, 1], false; "another shape")]
#[test_case(&[1, 704, 512, 512], &[1, 0], false; "another candidate order")]
fn keys_are_stable(shape: &[usize], candidates: &[usize], same: bool) {
    let base = key(&[1, 704, 512, 512], &[0, 1]);
    let other = key(shape, candidates);
    assert_eq!(base.line(Kind::Choice, 3) == other.line(Kind::Choice, 3), same);
    let line = base.line(Kind::Choice, 3);
    assert!(line.starts_with("choice|gemm|sm_86-28sm|BFloat16|1x704x512x512|"), "{line}");
    assert_ne!(line, base.line(Kind::Choice, 4), "the program fingerprint is part of the line");
    assert_ne!(line, base.line(Kind::Admitted, 3), "the kind is part of the line");
}

/// Two candidates of one shape run on the same scratch buffers, parameter by
/// parameter; a program's two parameters of one size never alias.
#[test]
fn candidates_share_scratch_buffers_without_aliasing_a_program() {
    let spec = |tile| GemmSpec {
        m: 256,
        n: 256,
        k: 256,
        batch: Batch::Static(1),
        epilogue: Epilogue::default(),
        cfg: GemmCfg { tile, stages: 2, warps: [2, 2], group_m: 8, unroll: true },
    };
    let (first, second) = (gemm::<BF16>(&spec([64, 64, 32])), gemm::<BF16>(&spec([128, 64, 32])));
    let pool = ScratchPool::default();
    let (a, b) = (pool.buffers(&first.params), pool.buffers(&second.params));
    let same = |x: &svod_tensor::Tensor, y: &svod_tensor::Tensor| std::sync::Arc::ptr_eq(&x.uop(), &y.uop());
    assert_eq!(a.len(), 3, "a, b and c");
    assert!(a.iter().zip(&b).all(|(x, y)| same(x, y)), "the second candidate reuses the first's buffers");
    assert!(!same(&a[0], &a[1]), "a and b are both [256·256] bf16 yet distinct");
    assert!(!same(&a[0], &a[2]) && !same(&a[1], &a[2]));
}

/// A kernel spilled when it reports scratch bytes; a backend reporting no
/// resources, or no scratch, admits.
#[test_case(&[] => false; "nothing reported")]
#[test_case(&[None] => false; "no scratch field")]
#[test_case(&[Some(0), None] => false; "no scratch used")]
#[test_case(&[Some(0), Some(16)] => true; "the merge fits, the kernel spills")]
fn a_kernel_with_scratch_spilled(scratch: &[Option<u32>]) -> bool {
    let resources: Vec<KernelResources> = scratch
        .iter()
        .map(|&scratch_bytes| KernelResources {
            vgprs: Some(128),
            sgprs: None,
            lds_bytes: 0,
            scratch_bytes,
            wave_size: 32,
            occupancy: None,
        })
        .collect();
    tune::spilled(&resources)
}

/// `SVOD_TK3_TUNE=0`'s override: off and back on.
#[test]
fn tuning_can_be_turned_off() {
    let before = tune::enabled();
    tune::set_enabled(false);
    assert!(!tune::enabled());
    tune::set_enabled(true);
    assert!(tune::enabled());
    tune::set_enabled(before);
}

/// A ranked list's admission is memoized for the process: a second ask
/// compiles nothing, and nothing compiled to judge is remembered as such.
#[test_case(Some(vec![0, 2]); "admitted")]
#[test_case(None; "nothing built")]
fn admission_is_memoized(found: Option<Vec<usize>>) {
    let store = TuneStore::at(None);
    let k = key(&[1, 2, 3], &[0, 1, 2]);
    let want = found.clone().map(Into::into);
    assert_eq!(store.admitted(&k, || 7, || found), want);
    assert_eq!(store.admitted(&k, || 7, || unreachable!("compiled twice")), want);
}

/// An admission round-trips through the store as an `admitted` line of its
/// indices, so a second store on the directory judges nothing; under other
/// programs (a lowering change) it judges again, and a judgement of nothing
/// is never stored.
#[test]
fn an_admission_round_trips_through_the_store() {
    let dir = scratch("admission");
    let k = key(&[1, 2, 3], &[0, 1, 2]);
    let store = TuneStore::at(Some(dir.clone()));
    assert_eq!(store.admitted(&key(&[9], &[0]), || 7, || None), None);
    assert!(store.entries(&k).is_empty());
    assert_eq!(store.admitted(&k, || 7, || Some(vec![3, 0, 5])).as_deref(), Some(&[3, 0, 5][..]));
    assert_eq!(store_file(&dir), format!("format {FORMAT}\n{} 3 0 5\n", k.line(Kind::Admitted, 7)));
    let again = TuneStore::at(Some(dir.clone()));
    assert_eq!(again.admitted(&k, || 7, || unreachable!("judged twice")).as_deref(), Some(&[3, 0, 5][..]));
    let relowered = TuneStore::at(Some(dir.clone()));
    assert_eq!(relowered.admitted(&k, || 8, || Some(vec![1])).as_deref(), Some(&[1][..]));
    assert_eq!(store_file(&dir).lines().count(), 3);
    let _ = std::fs::remove_dir_all(dir);
}

/// A file of another line format is dropped whole and the next write starts
/// it over in this one; a file of this format, or without a format line
/// (the first format's), reads as it did and stays as it is until a write.
#[test_case(Some(FORMAT + 1), false; "another format")]
#[test_case(Some(FORMAT), true; "this format")]
#[test_case(None, true; "no format line")]
fn a_file_of_another_format_is_dropped(written: Option<u32>, kept: bool) {
    let dir = scratch(&format!("format-{written:?}"));
    let k = key(&[2, 2, 2], &[0, 1]);
    let path = dir.join(format!("sm_86_28sm-v{}.txt", env!("CARGO_PKG_VERSION")));
    let line = format!("{} 1 100", k.line(Kind::Choice, 7));
    let header = written.map_or(String::new(), |n| format!("format {n}\n"));
    std::fs::write(&path, format!("{header}{line}\n")).unwrap();
    let store = TuneStore::at(Some(dir.clone()));
    assert_eq!(store.entries(&k).len(), usize::from(kept));
    let measured = Cell::new(false);
    assert_eq!(store.select(&k, 2, || 7, timed(&[Some(9), Some(4)], &measured)), Some(1));
    assert_eq!(measured.get(), !kept);
    let want =
        if kept { format!("{header}{line}\n") } else { format!("format {FORMAT}\n{} 1 4\n", k.line(Kind::Choice, 7)) };
    assert_eq!(store_file(&dir), want);
    let _ = std::fs::remove_dir_all(dir);
}

/// Tunes a Nemotron GEMM on the device into a scratch directory; a second
/// store on that directory answers from disk without measuring.
#[test]
fn a_gemm_tunes_on_the_device() {
    let spec = svod_dtype::default_device::default_device();
    let Some(planner) = super::device_planner().filter(|_| matches!(spec, svod_dtype::DeviceSpec::Cuda { .. })) else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let device = &planner.device;
    let (m, n, k) = (704, 512, 512);
    let cfgs = super::first_eight(&planner.gemm_ranked(1, m, n, k, false));
    let batch = Batch::Var { name: "b".into(), min: 1, max: 1 };
    let build = |cfg: GemmCfg| {
        let spec = GemmSpec { m, n, k, batch: batch.clone(), epilogue: Epilogue::default(), cfg };
        vec![(gemm::<BF16>(&spec), cfg.lowering(device))]
    };
    let key = TuneKey::new("gemm", &device.target, tune::compiler(&spec), ScalarDType::BFloat16, &[1, m, n, k], &cfgs);
    let dir = scratch("device");
    let start = std::time::Instant::now();
    let chosen = TuneStore::at(Some(dir.clone())).choose(&key, &cfgs, build, &mut Bench::default());
    let chosen = cfgs[chosen.expect("measured")];
    eprintln!("tuned {} candidates in {:.1} s: {chosen:?}", cfgs.len(), start.elapsed().as_secs_f64());
    let programs = || fingerprint(&cfgs.iter().map(|&c| build(c)).collect::<Vec<_>>());
    let hit = TuneStore::at(Some(dir.clone())).select(&key, cfgs.len(), programs, || panic!("measured twice"));
    assert_eq!(hit.map(|i| cfgs[i]), Some(chosen));
    let _ = std::fs::remove_dir_all(dir);
}

/// A candidate's time is its fastest timed run: a run the device did not
/// stamp is skipped, not taken as the candidate failing, and one under half
/// the median of its runs (a stamp the device lost the middle of) is dropped.
#[test_case(&[Some(5), None, Some(3)] => Some(3); "a lost sample")]
#[test_case(&[None, Some(7)] => Some(7); "the first lost")]
#[test_case(&[None, None] => None; "none timed")]
#[test_case(&[Some(90_000), Some(1), Some(91_000), Some(90_500), Some(92_000)] => Some(90_000); "a bogus stamp")]
#[test_case(&[Some(100), Some(60), Some(110)] => Some(60); "a fast run within half the median")]
#[test_case(&[Some(100), Some(49), Some(110)] => Some(100); "past half the median")]
fn a_candidate_keeps_its_timed_runs(runs: &[Option<u64>]) -> Option<u64> {
    let runs = runs.iter().map(|r| r.map(std::time::Duration::from_nanos));
    tune::best(runs).map(|d| d.as_nanos() as u64)
}

/// Candidates compile on workers that see the caller's default device (a
/// thread-local): a caller pinning one without `SVOD_DEVICE` builds them
/// for it, not for the CPU.
#[test]
fn workers_see_the_callers_device() {
    use svod_dtype::DeviceSpec;
    use svod_dtype::default_device::{default_device, with_default_device};
    let pinned = DeviceSpec::Amd { device_id: 3 };
    let seen = with_default_device(pinned.clone(), || tune::in_parallel((0..64).collect(), |_: i32| default_device()));
    assert!(seen.iter().all(|d| *d == pinned), "{seen:?}");
}

/// The timed rounds: every candidate runs once, then only those within
/// [`CONTENDER`](tune::CONTENDER) of the first round's best run the rest
/// ([`ROUNDS`] in all); a candidate that fails (a run of 0 here, its list
/// run out) is excluded for good, whichever round it fails in; nothing
/// running leaves every candidate `None`. A candidate's list repeats its
/// last run.
#[test_case(&[&[100], &[150], &[110], &[500]], &[4, 1, 4, 1], &[Some(100), Some(150), Some(110), Some(500)]; "losers run once")]
#[test_case(&[&[100, 90, 95, 92], &[118, 121]], &[4, 2], &[Some(90), Some(118)]; "past the bound a contender stops, its minimum standing")]
#[test_case(&[&[100], &[]], &[4, 1], &[Some(100), None]; "a failure in the first round")]
#[test_case(&[&[100], &[105, 104, 0]], &[4, 3], &[Some(100), None]; "a failure in a later round")]
#[test_case(&[&[], &[]], &[1, 1], &[None, None]; "nothing runs")]
#[test_case(&[], &[], &[]; "no candidates")]
fn contenders_run_the_later_rounds(runs: &[&[u64]], want_calls: &[usize], want: &[Option<u64>]) {
    use std::cell::RefCell;
    let calls = RefCell::new(vec![0usize; runs.len()]);
    let got = tune::rounds(runs.len(), |i| {
        let mut calls = calls.borrow_mut();
        let run = runs[i].get(calls[i]).or(runs[i].last()).copied();
        calls[i] += 1;
        run.filter(|&t| t > 0).map(Duration::from_nanos)
    });
    assert_eq!(got.iter().map(|t| t.map(|t| t.as_nanos() as u64)).collect::<Vec<_>>(), want);
    assert_eq!(*calls.borrow(), want_calls);
}

/// A candidate's plan is prepared for its measurement alone: two scratch
/// plans of one program hold two compiled kernels, neither published for
/// the other, while the same program launched as an op is compiled once.
#[test]
fn a_scratch_plan_publishes_nothing() {
    let Some(planner) = super::device_planner() else {
        eprintln!("skipped: no GPU with tk3 tables");
        return;
    };
    let device = &planner.device;
    let cfg = super::first_eight(&planner.gemm_ranked(1, 704, 512, 512, false))[0];
    let spec = GemmSpec { m: 704, n: 512, k: 512, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg };
    let pool = ScratchPool::default();
    let scratch = || tune::scratch_plan(gemm::<BF16>(&spec), &cfg.lowering(device), &pool).expect("a plan");
    let kernels = |plan: &svod_runtime::ExecutionPlan| plan.kernels().map(std::ptr::from_ref).collect::<Vec<_>>();
    let (first, second) = (scratch(), scratch());
    assert!(!kernels(&first.0).is_empty());
    assert!(kernels(&first.0).iter().zip(kernels(&second.0)).all(|(a, b)| !std::ptr::eq(*a, b)));
    let launched = || {
        let buffers = pool.buffers(&gemm::<BF16>(&spec).params);
        let tensors: Vec<&svod_tensor::Tensor> = buffers.iter().collect();
        let out = crate::launch::graph_launch(gemm::<BF16>(&spec), &cfg.lowering(device), &tensors);
        out.expect("a launch").prepare().expect("a plan")
    };
    let (first, second) = (launched(), launched());
    assert!(kernels(&first).iter().zip(kernels(&second)).all(|(a, b)| std::ptr::eq(*a, b)));
}
