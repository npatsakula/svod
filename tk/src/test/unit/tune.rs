//! The tune store: memo, on-disk round trip of choices and values, key
//! stability, the rule that nothing unmeasured is stored, a key measured
//! once under concurrent asks, a store hit compiling nothing, a run's
//! stamps and the lost-stamp guard, what counts as a spill, the admission
//! memo, a fingerprint keyed by what the candidates emit and remembered
//! by what they are made of, and on the device a real GEMM tuning.

use super::Scratch;
use crate::ops::config::{Tiled, gemm_cfg};
use std::cell::Cell;
use std::time::Duration;

use svod_dtype::ScalarDType;
use svod_runtime::KernelResources;
use test_case::test_case;

use crate::atoms::{Device, sm86};
use crate::build::BF16;
use crate::kernels::Batch;
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::tune::{self, Bench, Entry, FORMAT, Kind, Pass, ScratchPool, TuneKey, TuneStore, fingerprint};

/// A host key: no device, so no compiler digest.
fn key(shape: &[usize], candidates: &[usize]) -> TuneKey {
    TuneKey::new("gemm", &sm86(), 0, ScalarDType::BFloat16, shape, &candidates)
}

/// `n` distinct GEMM configs a store line's indices point into.
fn cfgs(n: usize) -> Vec<crate::kernels::gemm::GemmCfg> {
    let tiles = [[64, 64, 32], [128, 64, 32], [64, 128, 32], [128, 128, 32], [256, 64, 16], [64, 256, 64]];
    (0..n)
        .map(|i| {
            let t = tiles[i % tiles.len()];
            gemm_cfg(t, 2, [2, 2], i % 2 == 0)
        })
        .chain(std::iter::repeat(gemm_cfg(tiles[0], 2, [2, 2], false)))
        .take(n)
        .collect()
}

/// Every family's config code holds the config whole and nothing else: the
/// lattices the rankings walk — warp and warpgroup cores, a two-crew
/// cluster, conv splits, nearest lists, lane queues and score matrices —
/// round-trip through their code, and a code that is not the family's form
/// holds no config.
#[test]
fn every_familys_config_code_round_trips() {
    use crate::atoms::Core;
    use crate::kernels::attention::FaCfg;
    use crate::kernels::conv::ConvCfg;
    use crate::kernels::nearest::{LaneQueue, Metric, NearestCfg, SelectCfg};
    use crate::kernels::rows::NormCfg;
    use crate::ops::config::{Tiled, gemm_lattice};
    fn round<C: Tiled + std::fmt::Debug>(cs: &[C]) {
        for c in cs {
            let code = c.code();
            assert_eq!(C::of_code(&code).as_ref(), Some(c), "{c:?} does not round-trip as {code}");
        }
    }
    let target = sm86();
    let planner = super::planner(target.clone());
    let mut gemm = gemm_lattice(&target);
    gemm.push(GemmCfg { core: Core::Group, ..gemm_cfg([128, 256, 32], 4, [4, 4], false) });
    round(&gemm);
    let mut conv: Vec<ConvCfg> = gemm.iter().map(|g| ConvCfg { gemm: *g, split: 4 }).collect();
    conv.push(ConvCfg { gemm: gemm[0], split: 1 });
    round(&conv);
    round(&[NormCfg { br: 4 }, NormCfg { br: 32 }]);
    let mut forms = planner.attention_ranked(8, 100, 100, 64, true, false).to_vec();
    forms.push(FaCfg { core: Core::Group, splits: 2, key_warps: 4, ..forms[0] });
    forms.push(FaCfg {
        forms: crate::kernels::attention::Forms {
            truncate_p: true,
            sum_after: true,
            skip_rescale: true,
            raw_max: true,
            bias_direct: true,
        },
        ..forms[0]
    });
    round(&forms);
    let problem = crate::ops::config::NearestProblem {
        n: 1 << 14,
        m: 1 << 14,
        d: 128,
        dtype: ScalarDType::Float16,
        metric: Metric::SquaredL2,
        floors: Default::default(),
        groups: false,
        k: 64,
    };
    let mut nearest = planner.nearest_ranked(problem).to_vec();
    nearest.push(NearestCfg { lanes: Some(LaneQueue { width: 4, depth: 32, passes: true }), ..nearest[0] });
    nearest.push(NearestCfg {
        select: Some(SelectCfg { warps: 2, queue: 16, filter: true }),
        lanes: None,
        ..nearest[0]
    });
    round(&nearest);
    // A code that is not the family's form holds no config.
    for bad in ["", "x", "w,1", "w,a,b,c,d,e,f,g,h", "64,64,2,0", "w,1,2,3,4,5,6,7,8,9,10"] {
        assert!(GemmCfg::of_code(bad).is_none());
        assert!(FaCfg::of_code(bad).is_none());
        assert!(NearestCfg::of_code(bad).is_none());
    }
}

/// The store file of [`key`]'s device under `dir`.
fn store_file(dir: &std::path::Path) -> String {
    let path = dir.join(format!("sm_86_28sm-v{}.txt", env!("CARGO_PKG_VERSION")));
    std::fs::read_to_string(path).expect("the store file")
}

/// A choice made the way [`TuneStore::choose`] makes one, from a closure of
/// device times rather than compiled candidates.
pub(super) trait Select {
    /// The winner among `count` candidates: the memo, then the store under
    /// the line `programs` fingerprints, else `measure` times every
    /// candidate (device ns, `None` where one cannot run) and their order is
    /// stored. `None` when nothing measured.
    fn select(
        &self,
        key: &TuneKey,
        count: usize,
        programs: impl FnOnce() -> u64,
        measure: impl FnOnce() -> Vec<Option<u64>>,
    ) -> Option<usize>;
}

impl Select for TuneStore {
    fn select(
        &self,
        key: &TuneKey,
        count: usize,
        programs: impl FnOnce() -> u64,
        measure: impl FnOnce() -> Vec<Option<u64>>,
    ) -> Option<usize> {
        let within = |entry: &Entry| matches!(entry, Entry::Choice(timed) if timed.iter().all(|&(i, _, _)| i < count));
        let timed = |()| {
            tune::ordered(measure()).map(|timed| {
                Entry::Choice(timed.iter().map(|&(i, ns)| (i, i.to_string(), ns)).collect::<Vec<_>>().into())
            })
        };
        match self.remember(Kind::Choice, key, within, || (programs(), ()), timed)? {
            Entry::Choice(timed) => Some(timed[0].0),
            _ => None,
        }
    }
}

/// A measurement that records it ran.
fn timed<'a>(ns: &'a [Option<u64>], ran: &'a Cell<bool>) -> impl FnOnce() -> Vec<Option<u64>> + 'a {
    move || {
        ran.set(true);
        ns.to_vec()
    }
}

/// The fastest is stored, its line holding the candidates that ran in timed
/// order with their times (one that did not run left out), so a later process
/// knows the leaders and how close they ran; a second store on the directory
/// reads it back without measuring; a kernel change (another program
/// fingerprint) re-measures.
/// A store line is its key, a space and the entry, the key's fields split
/// at bars: an op name with either would never be read back.
#[test_case("packed attention"; "a space")]
#[test_case("attention|packed"; "a bar")]
#[should_panic(expected = "a store line is split at spaces and bars")]
fn an_op_name_splits_no_store_line(op: &'static str) {
    TuneKey::new(op, &sm86(), 0, ScalarDType::BFloat16, &[1], &[0usize]);
}

#[test]
fn a_winner_round_trips_through_the_store() {
    let dir = Scratch::new("tune-round-trip");
    let k = key(&[1, 704, 512, 512], &[1, 2, 3]);
    let ran = Cell::new(false);
    let chosen = TuneStore::at(Some(dir.to_path_buf())).select(&k, 3, || 7, timed(&[Some(300), None, Some(100)], &ran));
    assert_eq!((chosen, ran.get()), (Some(2), true));
    assert_eq!(store_file(&dir), format!("format {FORMAT}\n{} 2:2 100 0:0 300\n", k.line(Kind::Choice, 7)));

    let again = TuneStore::at(Some(dir.to_path_buf()));
    let ran = Cell::new(false);
    assert_eq!((again.select(&k, 3, || 7, timed(&[None; 3], &ran)), ran.get()), (Some(2), false));

    let changed = TuneStore::at(Some(dir.to_path_buf()));
    let ran = Cell::new(false);
    assert_eq!(changed.select(&k, 3, || 8, timed(&[Some(5), Some(9), Some(9)], &ran)), Some(0));
    assert!(ran.get(), "another kernel fingerprint is a miss");
    assert_eq!(changed.entries(&k).len(), 2, "both lines are kept");
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
    let dir = Scratch::new("tune-unmeasured");
    let store = TuneStore::at(Some(dir.to_path_buf()));
    let k = key(&[64, 64, 64], &[0, 1]);
    assert_eq!(store.select(&k, 2, || 1, || vec![None, None]), None);
    assert!(store.entries(&k).is_empty());
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

/// Threads storing other keys of one device file at once lose none of
/// them: each write re-reads, merges and replaces the file alone.
#[test]
fn concurrent_writers_of_one_file_lose_no_line() {
    let (threads, keys) = (8, 8);
    let dir = Scratch::new("tune-concurrent");
    let store = TuneStore::at(Some(dir.to_path_buf()));
    let start = std::sync::Barrier::new(threads);
    std::thread::scope(|s| {
        for t in 0..threads {
            let (store, start) = (&store, &start);
            s.spawn(move || {
                start.wait();
                for i in 0..keys {
                    let k = TuneKey { op: "probe:step", ..key(&[t, i], &[0]) };
                    store.value(&k, || 1, || Some((t * keys + i) as u64));
                }
            });
        }
    });
    let again = TuneStore::at(Some(dir.to_path_buf()));
    assert_eq!(again.entries(&key(&[], &[0])).len(), threads * keys);
    for t in 0..threads {
        for i in 0..keys {
            let k = TuneKey { op: "probe:step", ..key(&[t, i], &[0]) };
            assert_eq!(again.value(&k, || 1, || panic!("lost")), Some((t * keys + i) as u64));
        }
    }
    let leftovers = std::fs::read_dir(&dir).unwrap().count();
    assert_eq!(leftovers, 1, "a temporary file left behind");
}

/// A plain bf16 GEMM's programs under `cfg` on sm_86.
fn gemm_programs(cfg: crate::kernels::gemm::GemmCfg) -> tune::Candidate {
    let spec = GemmSpec { m: 128, n: 128, k: 128, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg };
    vec![(gemm::<BF16>(&spec), cfg.lowering(&Device { target: sm86(), costs: super::costs() }))]
}

/// A choice keys by the candidates' programs, each built once: a store
/// miss compiles each candidate once (and times them where the host's
/// device runs sm_86 programs), and a store hit in another process builds
/// the programs to fingerprint them and compiles nothing.
#[test]
fn a_store_hit_compiles_nothing() {
    let dir = Scratch::new("tune-hit");
    let k = key(&[128, 128, 128], &[0, 1]);
    let cfgs = [gemm_cfg([64, 64, 32], 2, [2, 2], true), gemm_cfg([128, 64, 32], 2, [2, 2], true)];
    let builds = Cell::new(0);
    let build = |c: crate::kernels::gemm::GemmCfg| {
        builds.set(builds.get() + 1);
        gemm_programs(c)
    };
    let programs = fingerprint(&[gemm_programs(cfgs[0]), gemm_programs(cfgs[1])]);
    let mut bench = Bench::default();
    let missed = TuneStore::at(Some(dir.to_path_buf())).choose(&k, &cfgs, build, &mut bench, Pass::Whole, None);
    assert_eq!((builds.get(), bench.built.len()), (2, 2), "a miss builds and compiles each once");
    // Without a device that timed them, what it would have measured, stored
    // as a timing would; with one, its choice is already stored.
    let stored = TuneStore::at(Some(dir.to_path_buf())).select(&k, 2, || programs, || vec![Some(9), Some(4)]);
    let missed = missed.map(|timed| timed[0].0);
    assert_eq!(stored, missed.or(Some(1)), "the device's choice, else the stored timing's");
    builds.set(0);
    let mut bench = Bench::default();
    let hit = TuneStore::at(Some(dir.to_path_buf())).choose(&k, &cfgs, build, &mut bench, Pass::Whole, None);
    assert_eq!((hit.map(|timed| timed[0].0), builds.get(), bench.built.len()), (stored, 2, 0));
}

/// A stored index past the candidate count is a miss.
#[test]
fn a_stale_index_is_a_miss() {
    let dir = Scratch::new("tune-stale");
    let k = key(&[8, 8, 8], &[0, 1, 2]);
    TuneStore::at(Some(dir.to_path_buf())).select(&k, 3, || 1, || vec![None, None, Some(1)]);
    let ran = Cell::new(false);
    assert_eq!(TuneStore::at(Some(dir.to_path_buf())).select(&k, 2, || 1, timed(&[Some(1), None], &ran)), Some(0));
    assert!(ran.get());
}

/// A warm memo answers a value without fingerprinting or measuring; an
/// unmeasured value is memoized as missing and never stored.
#[test_case(Some(5); "measured")]
#[test_case(None; "unmeasured")]
fn the_memo_answers_a_value(measured: Option<u64>) {
    let dir = Scratch::new(&format!("tune-value-memo-{measured:?}"));
    let store = TuneStore::at(Some(dir.to_path_buf()));
    let k = key(&[1], &[0]);
    assert_eq!(store.value(&k, || 1, || measured), measured);
    assert_eq!(store.entries(&k).len(), usize::from(measured.is_some()));
    assert_eq!(store.value(&k, || unreachable!("fingerprinted"), || unreachable!("measured")), measured);
}

/// Choices and values of one device share its file, a value as a `value`
/// line holding the number; writing one keeps the other, and a second store
/// on the directory reads both back without measuring.
#[test]
fn choices_and_values_share_a_file() {
    let dir = Scratch::new("tune-shared");
    let (choice, value) = (key(&[64, 64, 64], &[0, 1]), TuneKey { op: "probe:step", ..key(&[], &[2]) });
    TuneStore::at(Some(dir.to_path_buf())).select(&choice, 2, || 1, || vec![Some(9), Some(4)]);
    assert_eq!(TuneStore::at(Some(dir.to_path_buf())).value(&value, || 2, || Some(1234)), Some(1234));
    let lines =
        [format!("{} 1:1 4 0:0 9", choice.line(Kind::Choice, 1)), format!("{} 1234", value.line(Kind::Value, 2))];
    assert_eq!(store_file(&dir), format!("format {FORMAT}\n{}\n", lines.join("\n")));
    let again = TuneStore::at(Some(dir.to_path_buf()));
    assert_eq!(again.entries(&choice).len(), 2);
    assert_eq!(again.select(&choice, 2, || 1, || panic!("measured twice")), Some(1));
    assert_eq!(again.value(&value, || 2, || panic!("measured twice")), Some(1234));
}

/// A line the store cannot read (an older format, a stray edit) is a miss
/// and is dropped at the next write; a line of another kind under the same
/// key is kept, each kind leading its own line.
#[test]
fn an_unreadable_line_is_dropped_at_the_next_write() {
    let dir = Scratch::new("tune-unreadable");
    let k = key(&[3, 3, 3], &[0, 1]);
    let path = dir.join(format!("sm_86_28sm-v{}.txt", env!("CARGO_PKG_VERSION")));
    let kept = format!("{} 42", k.line(Kind::Value, 1));
    let lines = format!("format {FORMAT}\ngemm|sm_86-28sm|BFloat16|3x3x3|0|0|0 1 100\n{kept}\nnonsense\n");
    std::fs::write(&path, lines).unwrap();
    let store = TuneStore::at(Some(dir.to_path_buf()));
    assert_eq!(store.entries(&k).len(), 1);
    assert_eq!(store.select(&k, 2, || 1, || vec![Some(9), Some(4)]), Some(1));
    assert_eq!(store_file(&dir), format!("format {FORMAT}\n{} 1:1 4 0:0 9\n{kept}\n", k.line(Kind::Choice, 1)));
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

/// A line keys the op, the device, the dtype, the shape, the candidates in
/// order, the compiler (toolchain, flags), the kind and the programs'
/// fingerprint: another of any is another line, so it re-measures.
#[test_case(&[1, 704, 512, 512], &[0, 1], 0, true; "the same inputs")]
#[test_case(&[1, 704, 512, 2048], &[0, 1], 0, false; "another shape")]
#[test_case(&[1, 704, 512, 512], &[1, 0], 0, false; "another candidate order")]
#[test_case(&[1, 704, 512, 512], &[0, 1], 1, false; "another compiler")]
fn keys_are_stable(shape: &[usize], candidates: &[usize], compiler: u64, same: bool) {
    let base = key(&[1, 704, 512, 512], &[0, 1]);
    let other = TuneKey { compiler: base.compiler ^ compiler, ..key(shape, candidates) };
    assert_eq!(base.line(Kind::Choice, 3) == other.line(Kind::Choice, 3), same);
    let line = base.line(Kind::Choice, 3);
    assert!(line.starts_with("choice|gemm|sm_86-28sm|BFloat16|1x704x512x512|"), "{line}");
    assert!(line.contains(&format!("|{:016x}|", base.compiler)), "{line}");
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
        cfg: gemm_cfg(tile, 2, [2, 2], true),
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
            agprs: None,
            sgprs: None,
            lds_bytes: 0,
            scratch_bytes,
            wave_size: 32,
            occupancy: None,
        })
        .collect();
    tune::spilled(&resources)
}

/// A scoped setting holds on its thread and on the workers `in_parallel`
/// runs there, nests, and gives the outer one back on return and on unwind;
/// another thread follows the environment.
#[test_case(false; "off")]
#[test_case(true; "on")]
fn a_scoped_setting_holds_on_its_thread_alone(on: bool) {
    let outer = tune::enabled();
    tune::scoped(on, || {
        assert_eq!(tune::enabled(), on);
        assert!(tune::in_parallel((0..64).collect(), |_: i32| tune::enabled()).into_iter().all(|e| e == on));
        assert_eq!(std::thread::spawn(tune::enabled).join().unwrap(), outer);
        tune::scoped(!on, || assert_eq!(tune::enabled(), !on));
        assert_eq!(tune::enabled(), on);
        assert!(std::panic::catch_unwind(|| tune::scoped(!on, || panic!("unwinds"))).is_err());
        assert_eq!(tune::enabled(), on);
    });
    assert_eq!(tune::enabled(), outer);
}

/// A ranked list's admission is memoized for the process: a second ask
/// compiles nothing, and nothing compiled to judge is remembered as such.
#[test_case(Some(vec![0, 2]); "admitted")]
#[test_case(None; "nothing built")]
fn admission_is_memoized(found: Option<Vec<usize>>) {
    let store = TuneStore::at(None);
    let (k, list) = (key(&[1, 2, 3], &[0, 1, 2]), cfgs(3));
    let want = found.clone().map(Into::into);
    assert_eq!(store.admitted(&k, &list, || 7, || found), want);
    assert_eq!(store.admitted(&k, &list, || 7, || unreachable!("compiled twice")), want);
}

/// An admission round-trips through the store as an `admitted` line of its
/// indices, so a second store on the directory judges nothing; under other
/// programs (a lowering change) it judges again, and a judgement of nothing
/// is never stored.
#[test]
fn an_admission_round_trips_through_the_store() {
    let dir = Scratch::new("tune-admission");
    let k = key(&[1, 2, 3], &[0, 1, 2]);
    let store = TuneStore::at(Some(dir.to_path_buf()));
    let list = cfgs(6);
    assert_eq!(store.admitted(&key(&[9], &[0]), &list, || 7, || None), None);
    assert!(store.entries(&k).is_empty());
    assert_eq!(store.admitted(&k, &list, || 7, || Some(vec![3, 0, 5])).as_deref(), Some(&[3, 0, 5][..]));
    let coded = |at: usize| format!("{at}:{}", list[at].code());
    assert_eq!(
        store_file(&dir),
        format!("format {FORMAT}\n{} {} {} {}\n", k.line(Kind::Admitted, 7), coded(3), coded(0), coded(5))
    );
    let again = TuneStore::at(Some(dir.to_path_buf()));
    assert_eq!(again.admitted(&k, &list, || 7, || unreachable!("judged twice")).as_deref(), Some(&[3, 0, 5][..]));
    let relowered = TuneStore::at(Some(dir.to_path_buf()));
    assert_eq!(relowered.admitted(&k, &list, || 8, || Some(vec![1])).as_deref(), Some(&[1][..]));
    assert_eq!(store_file(&dir).lines().count(), 3);
}

/// A first clean config round-trips through the store as a `first` line
/// beside the list's admission, each remembered once: a second store on
/// the directory searches nothing, and a search that found nothing is not
/// stored.
#[test]
fn a_first_clean_config_round_trips_through_the_store() {
    let dir = Scratch::new("tune-first");
    let k = key(&[1, 2, 3], &[0, 1, 2]);
    let store = TuneStore::at(Some(dir.to_path_buf()));
    let list = cfgs(6);
    assert_eq!(store.first(&key(&[9], &[0]), &list, || 7, || None), None);
    assert!(store.entries(&k).is_empty());
    assert_eq!(store.first(&k, &list, || 7, || Some(2)), Some(2));
    assert_eq!(store.admitted(&k, &list, || 7, || Some(vec![2, 0, 5])).as_deref(), Some(&[2, 0, 5][..]));
    let lines = format!(
        "format {FORMAT}\n{} 2:{} 0:{} 5:{}\n{} 2:{}\n",
        k.line(Kind::Admitted, 7),
        list[2].code(),
        list[0].code(),
        list[5].code(),
        k.line(Kind::First, 7),
        list[2].code()
    );
    assert_eq!(store_file(&dir), lines);
    let again = TuneStore::at(Some(dir.to_path_buf()));
    assert_eq!(again.first(&k, &list, || 7, || unreachable!("searched twice")), Some(2));
}

/// A file of another line format, or without a format line (the first
/// format's), is dropped whole and the next write starts it over in this
/// one; a file of this format reads as it did and stays as it is until a write.
#[test_case(Some(FORMAT + 1), false; "another format")]
#[test_case(Some(FORMAT), true; "this format")]
#[test_case(None, false; "no format line")]
fn a_file_of_another_format_is_dropped(written: Option<u32>, kept: bool) {
    let dir = Scratch::new(&format!("tune-format-{written:?}"));
    let k = key(&[2, 2, 2], &[0, 1]);
    let path = dir.join(format!("sm_86_28sm-v{}.txt", env!("CARGO_PKG_VERSION")));
    let line = format!("{} 1:1 100 0:0 300", k.line(Kind::Choice, 7));
    let header = written.map_or(String::new(), |n| format!("format {n}\n"));
    std::fs::write(&path, format!("{header}{line}\n")).unwrap();
    let store = TuneStore::at(Some(dir.to_path_buf()));
    assert_eq!(store.entries(&k).len(), usize::from(kept));
    let measured = Cell::new(false);
    assert_eq!(store.select(&k, 2, || 7, timed(&[Some(9), Some(4)], &measured)), Some(1));
    assert_eq!(measured.get(), !kept);
    let want = if kept {
        format!("{header}{line}\n")
    } else {
        format!("format {FORMAT}\n{} 1:1 4 0:0 9\n", k.line(Kind::Choice, 7))
    };
    assert_eq!(store_file(&dir), want);
}

/// Tunes a Nemotron GEMM on the device into a scratch directory; a second
/// store on that directory answers from disk without measuring.
#[test]
fn a_gemm_tunes_on_the_device() {
    let Some(planner) = super::device_planner() else { return };
    let spec = svod_dtype::default_device::default_device();
    let device = &planner.device;
    let (m, n, k) = (704, 512, 512);
    let cfgs = super::first_eight(&planner.gemm_ranked(1, m, n, k, false));
    let batch = Batch::var("b", 1, 1);
    let build = |cfg: GemmCfg| {
        let spec = GemmSpec { m, n, k, batch: batch.clone(), epilogue: Epilogue::default(), cfg };
        vec![(gemm::<BF16>(&spec), cfg.lowering(device))]
    };
    let key = TuneKey::new("gemm", &device.target, tune::compiler(&spec), ScalarDType::BFloat16, &[1, m, n, k], &cfgs);
    let dir = Scratch::new("tune-device");
    let start = std::time::Instant::now();
    let chosen =
        TuneStore::at(Some(dir.to_path_buf())).choose(&key, &cfgs, build, &mut Bench::default(), Pass::Whole, None);
    let chosen = cfgs[chosen.expect("measured")[0].0];
    eprintln!("tuned {} candidates in {:.1} s: {chosen:?}", cfgs.len(), start.elapsed().as_secs_f64());
    let programs = || fingerprint(&cfgs.iter().map(|&c| build(c)).collect::<Vec<_>>());
    let hit = TuneStore::at(Some(dir.to_path_buf())).select(&key, cfgs.len(), programs, || panic!("measured twice"));
    assert_eq!(hit.map(|i| cfgs[i]), Some(chosen));
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
/// ([`ROUNDS`] in all), past the second only those within
/// [`SETTLED`](tune::SETTLED) of the best so far, or every candidate every
/// round when there are fewer than [`CUT_FROM`](tune::CUT_FROM) (`losers`
/// more at 1000 fill a field out); then, where more than two contenders ran
/// them, the two leaders run [`ROUNDS`] more.
/// A candidate that fails (a run of 0 here, its list run out) is excluded
/// for good, whichever round it fails in; nothing running leaves every
/// candidate `None`. A candidate's list repeats its last run.
#[test_case(&[&[100], &[150], &[110], &[500]], 12, &[4, 1, 2, 1], &[Some(100), Some(150), Some(110), Some(500)]; "in a field losers run once and an unsettled contender twice")]
#[test_case(&[&[100], &[150], &[110], &[500]], 4, &[4, 1, 2, 1], &[Some(100), Some(150), Some(110), Some(500)]; "eight candidates are a field")]
#[test_case(&[&[100], &[115], &[110], &[500]], 12, &[8, 2, 6, 1], &[Some(100), Some(115), Some(110), Some(500)]; "in a field the leaders of three contenders run again")]
#[test_case(&[&[100], &[104], &[107], &[103, 103, 98]], 12, &[8, 3, 2, 8], &[Some(100), Some(104), Some(107), Some(98)]; "past the second round the settled contenders run on")]
#[test_case(&[&[100], &[110, 104]], 14, &[4, 4], &[Some(100), Some(104)]; "a contender settles by its best after the round")]
#[test_case(&[&[100], &[150], &[110], &[500]], 0, &[8, 4, 8, 4], &[Some(100), Some(150), Some(110), Some(500)]; "a short list runs every round")]
#[test_case(&[&[100, 90, 95, 92], &[118, 121]], 14, &[4, 2], &[Some(90), Some(118)]; "past the bound a contender stops, its minimum standing")]
#[test_case(&[&[100; 4], &[101; 4], &[150]], 0, &[8, 8, 4], &[Some(100), Some(101), Some(150)]; "the two leaders run again")]
#[test_case(&[&[100, 100, 100, 100, 95], &[101, 101, 101, 101, 94], &[150]], 0, &[8, 8, 4], &[Some(95), Some(94), Some(150)]; "the leaders' rounds decide")]
#[test_case(&[&[100, 100, 100, 100, 95], &[101, 101, 101, 101, 94]], 0, &[4, 4], &[Some(100), Some(101)]; "a pair has run together already")]
#[test_case(&[&[100], &[]], 0, &[4, 1], &[Some(100), None]; "a failure in the first round")]
#[test_case(&[&[100], &[105, 104, 0]], 0, &[4, 3], &[Some(100), None]; "a failure in a later round")]
#[test_case(&[&[100, 100, 100, 100, 0], &[101], &[150]], 0, &[5, 8, 4], &[None, Some(101), Some(150)]; "a failure in the leaders' rounds")]
#[test_case(&[&[], &[]], 0, &[1, 1], &[None, None]; "nothing runs")]
#[test_case(&[], 0, &[], &[]; "no candidates")]
fn contenders_run_the_later_rounds(runs: &[&[u64]], losers: usize, want_calls: &[usize], want: &[Option<u64>]) {
    scripted_rounds(runs, losers, Pass::Whole, want_calls, want);
}

/// [`tune::rounds`] under `pass` over candidates whose runs are scripted
/// (`losers` more at 1000), checked against the times and the calls each
/// gets, every loser's one.
fn scripted_rounds(runs: &[&[u64]], losers: usize, pass: Pass, want_calls: &[usize], want: &[Option<u64>]) {
    use std::cell::RefCell;
    let runs: Vec<&[u64]> = runs.iter().copied().chain(std::iter::repeat_n(&[1000][..], losers)).collect();
    let calls = RefCell::new(vec![0usize; runs.len()]);
    let got = tune::rounds(runs.len(), None, pass, |i, _, _| {
        let mut calls = calls.borrow_mut();
        let run = runs[i].get(calls[i]).or(runs[i].last()).copied();
        calls[i] += 1;
        run.filter(|&t| t > 0).map(Duration::from_nanos)
    });
    let got: Vec<Option<u64>> = got.iter().map(|t| t.map(|t| t.as_nanos() as u64)).collect();
    assert_eq!(got[..want.len()], *want);
    assert!(got[want.len()..].iter().all(|t| *t == Some(1000)), "{got:?}");
    let calls = calls.borrow();
    assert_eq!(calls[..want_calls.len()], *want_calls);
    assert!(calls[want_calls.len()..].iter().all(|&c| c == 1), "{calls:?}");
}

/// The first round runs in the given order and the later ones backwards and
/// forwards in turn, so each candidate runs after both its neighbours; the
/// leaders' rounds alternate the two, the only calls told they are a
/// leaders' round. A screen's second round runs backwards.
#[test_case(Pass::Whole, &[0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0], &[0, 1, 0, 1, 0, 1, 0, 1]; "every round")]
#[test_case(Pass::Screen, &[0, 1, 2, 3, 3, 2, 1, 0], &[]; "a screen")]
fn later_rounds_alternate_their_order(pass: Pass, want: &[usize], leaders: &[usize]) {
    let mut order = vec![];
    tune::rounds(4, None, pass, |i, _, leader| {
        order.push((i, leader));
        Some(Duration::from_nanos(100 + i as u64))
    });
    let want: Vec<(usize, bool)> = want.iter().map(|&i| (i, false)).chain(leaders.iter().map(|&i| (i, true))).collect();
    assert_eq!(order, want);
}

/// A screen runs the first round and one more of its contenders, and no
/// leaders' rounds: a later pass times its leaders again. A short list's
/// screen runs every candidate twice.
#[test_case(&[&[100], &[150], &[110], &[500]], 12, &[2, 1, 2, 1], &[Some(100), Some(150), Some(110), Some(500)]; "a field")]
#[test_case(&[&[100], &[103, 101], &[104], &[500]], 12, &[2, 2, 2, 1], &[Some(100), Some(101), Some(104), Some(500)]; "a field of settled contenders")]
#[test_case(&[&[100], &[150], &[110], &[500]], 0, &[2, 2, 2, 2], &[Some(100), Some(150), Some(110), Some(500)]; "a short list")]
#[test_case(&[&[100], &[105, 0]], 0, &[2, 2], &[Some(100), None]; "a failure in the second round")]
fn a_screen_runs_two_rounds(runs: &[&[u64]], losers: usize, want_calls: &[usize], want: &[Option<u64>]) {
    scripted_rounds(runs, losers, Pass::Screen, want_calls, want);
}

/// Under a hopeless factor of 2, a candidate's first run past twice the
/// best time so far is its time (its settled runs skipped, `aborted`), and
/// a short list's later rounds leave out what sits past twice the first
/// round's best: candidates are `(first run, settled)`, the first
/// candidate is never cut short (there is no time before it), the bar is
/// the best so far, not the first candidate's, and a run at the bar runs on.
#[test_case(&[(100, 100), (300, 250), (110, 105)], &[4, 1, 4], &[false, true, false], &[100, 300, 105]; "a slow first run stops")]
#[test_case(&[(500, 500), (100, 100)], &[1, 4], &[false, false], &[500, 100]; "the first candidate runs whole")]
#[test_case(&[(100, 100), (150, 90), (190, 190)], &[4, 4, 1], &[false, false, true], &[100, 90, 190]; "the bar is the best so far")]
#[test_case(&[(100, 100), (200, 199)], &[4, 4], &[false, false], &[100, 199]; "a run at the bar runs on")]
#[test_case(&[(100, 100), (150, 250)], &[4, 1], &[false, false], &[100, 250]; "a settled time past the bar runs no later round")]
fn hopeless_candidates_stop_at_their_first_run(
    runs: &[(u64, u64)],
    want_calls: &[usize],
    want_aborted: &[bool],
    want: &[u64],
) {
    use std::cell::RefCell;
    let (calls, aborted) = (RefCell::new(vec![0usize; runs.len()]), RefCell::new(vec![false; runs.len()]));
    let got = tune::rounds(runs.len(), Some(2.0), Pass::Whole, |i, bar, _| {
        let (first, settled) = runs[i];
        let mut calls = calls.borrow_mut();
        calls[i] += 1;
        if calls[i] == 1 && bar.is_some_and(|bar| Duration::from_nanos(first) > bar) {
            aborted.borrow_mut()[i] = true;
            return Some(Duration::from_nanos(first));
        }
        Some(Duration::from_nanos(settled))
    });
    let got: Vec<u64> = got.iter().map(|t| t.expect("a time").as_nanos() as u64).collect();
    assert_eq!(got, want);
    assert_eq!(*calls.borrow(), want_calls);
    assert_eq!(*aborted.borrow(), want_aborted);
}

/// Without a hopeless factor nothing is cut short and a short list runs
/// every round, however far behind (the probes' pairs, a probe at least
/// twice its baseline, are timed so).
#[test]
fn without_a_hopeless_factor_every_candidate_runs_whole() {
    let mut calls = vec![0; 2];
    let got = tune::rounds(2, None, Pass::Whole, |i, bar, _| {
        assert_eq!(bar, None);
        calls[i] += 1;
        Some(Duration::from_nanos([100, 1000][i]))
    });
    assert_eq!(got, [Some(Duration::from_nanos(100)), Some(Duration::from_nanos(1000))]);
    assert_eq!(calls, [4, 4]);
}

/// A candidate's plan is prepared for its measurement alone: two scratch
/// plans of one program hold two compiled kernels, neither published for
/// the other, while the same program launched as an op is compiled once.
#[test]
fn a_scratch_plan_publishes_nothing() {
    let Some(planner) = super::device_planner() else { return };
    let device = &planner.device;
    let cfg = super::first_eight(&planner.gemm_ranked(1, 704, 512, 512, false))[0];
    let spec = GemmSpec { m: 704, n: 512, k: 512, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg };
    let pool = ScratchPool::default();
    let scratch = || tune::scratch_plan(gemm::<BF16>(&spec), &cfg.lowering(device), &pool).expect("a plan");
    let kernels = |plan: &svod_runtime::ExecutionPlan| plan.kernels().map(std::ptr::from_ref).collect::<Vec<_>>();
    let (first, second) = (scratch(), scratch());
    assert!(!kernels(&first.0).is_empty());
    assert!(kernels(&first.0).iter().zip(kernels(&second.0)).all(|(a, b)| !std::ptr::eq(*a, b)));
    let names = |plan: &svod_runtime::ExecutionPlan| plan.kernels().map(|k| k.entry_point.clone()).collect::<Vec<_>>();
    assert_eq!(names(&first.0), names(&second.0), "a candidate's kernels are named by their shape alone");
    let launched = || {
        let buffers = pool.buffers(&gemm::<BF16>(&spec).params);
        let tensors: Vec<&svod_tensor::Tensor> = buffers.iter().collect();
        let out = crate::launch::graph_launch(gemm::<BF16>(&spec), &cfg.lowering(device), &tensors);
        out.expect("a launch").prepare().expect("a plan")
    };
    let (first, second) = (launched(), launched());
    assert!(kernels(&first).iter().zip(kernels(&second)).all(|(a, b)| std::ptr::eq(*a, b)));
}

/// The pool's fill is prepared as a candidate is, in passing: a kernel of
/// the fill's shape prepared after it, as a model's are after a tune, gets
/// the bare shape name an ephemeral prepare keeps, not the counter's next
/// (Nemotron's first warm run after a tune compiled its elementwise kernels
/// again, under `n1` names).
#[test]
fn a_scratch_fill_moves_no_kernel_name_counter() {
    use crate::ir::{Param, ParamKind};
    use svod_dtype::DType;
    use svod_tensor::{PrepareConfig, Tensor};
    // A prime count: no other kernel of the process has the fill's shape.
    let elems = 999_983;
    let names = |plan: &svod_runtime::ExecutionPlan| plan.kernels().map(|k| k.entry_point.clone()).collect::<Vec<_>>();
    let mut ephemeral = PrepareConfig::from_env();
    ephemeral.ephemeral = true;
    // A tensor prepares once; the ephemeral prepares publish nothing the
    // later one could read.
    let bare = |t: Tensor| names(&t.prepare_with(&ephemeral).expect("a plan"));
    let (zeros, ones) = (
        || Tensor::zeros(&[elems], DType::Float32).contiguous(),
        || Tensor::ones(&[elems], DType::Float32).contiguous(),
    );
    let expected = bare(ones());
    assert!(!expected.is_empty());
    assert_eq!(bare(zeros()), expected, "the fill and the kernel after it share a shape name");
    let pool = ScratchPool::default();
    let param = Param { name: "x".into(), dtype: ScalarDType::Float32, kind: ParamKind::In, elems };
    assert_eq!(pool.buffers(&[param]).len(), 1);
    assert_eq!(names(&ones().prepare().expect("a plan")), expected, "the fill moved the counter");
}

/// A float scratch buffer holds values spread over `[-1, 1)`, an integer one
/// zeros: the first toggles bits as real data does, the second is read as
/// the slot a probe reads next.
#[test]
fn a_float_scratch_buffer_is_random_and_an_integer_one_is_zero() {
    use crate::ir::{Param, ParamKind};
    let pool = ScratchPool::default();
    let param = |dtype| Param { name: "x".into(), dtype, kind: ParamKind::In, elems: 4096 };
    let [floats, ints] =
        pool.buffers(&[param(ScalarDType::Float32), param(ScalarDType::Int32)]).try_into().expect("two");
    let floats: Vec<f32> = floats.to_vec().expect("values");
    let ints: Vec<i32> = ints.to_vec().expect("values");
    assert!(floats.iter().all(|x| (-1.0..1.0).contains(x)), "values in [-1, 1)");
    let (negative, positive) =
        (floats.iter().filter(|&&x| x < -0.5).count(), floats.iter().filter(|&&x| x > 0.5).count());
    assert!(negative > 512 && positive > 512, "spread over the range: {negative} below -0.5, {positive} above 0.5");
    assert!(ints.iter().all(|&x| x == 0), "an integer parameter is zero");
}

/// A 4096³ GEMM candidate on the warp-level core for `target` under `costs`.
fn gemm_candidate(target: crate::atoms::Target, costs: crate::atoms::Costs) -> tune::Candidate {
    let cfg = crate::ops::config::gemm_cfg([128, 128, 32], 2, [2, 4], true);
    let spec = GemmSpec { m: 4096, n: 4096, k: 4096, batch: Batch::Static(1), epilogue: Epilogue::DEFAULT, cfg };
    vec![(gemm::<BF16>(&spec), cfg.lowering(&Device { target, costs }))]
}

/// The fingerprint keys the emitted kernel: a target field (the waves a
/// SIMD schedules, read by the ranking alone) or a cost (the memory
/// latency) no instruction reads leaves it, a field the lowering reads (the
/// dies the blocks are renumbered over) moves it.
#[test]
fn the_fingerprint_keys_what_is_emitted() {
    let fingerprint = |c| TuneStore::at(None).fingerprint(&[c]);
    let (target, costs) = (crate::atoms::Target::card("mi300x").expect("a listed card"), super::costs());
    let base = fingerprint(gemm_candidate(target.clone(), costs));
    let occupancy = crate::atoms::Occupancy { waves: 2 * target.occupancy.waves, ..target.occupancy };
    let planned = crate::atoms::Target { occupancy, ..target.clone() };
    assert_eq!(fingerprint(gemm_candidate(planned, costs)), base, "a field only the ranking reads");
    let latency = crate::atoms::Costs { latency: 2.0 * costs.latency, ..costs };
    assert_eq!(fingerprint(gemm_candidate(target.clone(), latency)), base, "a cost only the ranking reads");
    let one_die = crate::atoms::Target { xcds: 1, ..target };
    assert_ne!(fingerprint(gemm_candidate(one_die, costs)), base, "the block order is emitted");
}

/// Candidates fingerprint as a list: their order and grouping count.
#[test]
fn the_fingerprint_keys_the_candidates_in_order() {
    let store = TuneStore::at(None);
    let (a, b) = (
        gemm_candidate(sm86(), super::costs()),
        gemm_candidate(crate::atoms::Target::card("mi300x").unwrap(), super::costs()),
    );
    let ab = store.fingerprint(&[a.clone(), b.clone()]);
    assert_eq!(TuneStore::at(None).fingerprint(&[a.clone(), b.clone()]), ab, "deterministic");
    assert_ne!(store.fingerprint(&[b.clone(), a.clone()]), ab);
    assert_ne!(store.fingerprint(&[[a, b].concat()]), ab);
}

/// The store under `dir` as a build named `build` sees it.
fn built(dir: &std::path::Path, build: u64) -> TuneStore {
    let mut store = TuneStore::at(Some(dir.to_path_buf()));
    store.build = Some(build);
    store
}

/// The digest of what a candidate's one program is made of, as the memo keys it.
fn made_of(candidate: &tune::Candidate) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    std::hash::Hash::hash(&format!("{:?}{:?}", candidate[0].0, candidate[0].1), &mut hasher);
    std::hash::Hasher::finish(&hasher)
}

/// What a program emits is kept in its build's file by what it is made
/// of: a later store of the build reads it rather than lowering, and a
/// line it cannot read is passed over.
#[test]
fn what_a_program_emits_is_remembered_by_its_build() {
    let dir = Scratch::new("tune-emitted");
    let candidate = gemm_candidate(sm86(), super::costs());
    let fresh = built(&dir, 1).fingerprint(std::slice::from_ref(&candidate));
    assert_eq!(fresh, TuneStore::at(None).fingerprint(std::slice::from_ref(&candidate)));
    let file = dir.join("emitted").join(format!("{:016x}.txt", 1));
    let text = std::fs::read_to_string(&file).expect("the build's file");
    assert!(text.starts_with(&format!("{:016x} ", made_of(&candidate))), "{text}");
    std::fs::write(&file, format!("not a line\n{:016x} 000000000000002a\n", made_of(&candidate))).expect("a file");
    let mut hasher = std::hash::DefaultHasher::new();
    std::hash::Hash::hash(&vec![vec![42u64]], &mut hasher);
    let read = built(&dir, 1).fingerprint(std::slice::from_ref(&candidate));
    assert_eq!(read, std::hash::Hasher::finish(&hasher), "read from the file, not lowered");
}

/// Another build consults none of a build's memo: it lowers again, and where
/// it emits the same code its fingerprint is the first build's, so the
/// store's entries still answer.
#[test]
fn another_build_lowers_again_and_matches_where_the_code_is_the_same() {
    let dir = Scratch::new("tune-rebuilt");
    let candidate = gemm_candidate(sm86(), super::costs());
    let first = built(&dir, 1).fingerprint(std::slice::from_ref(&candidate));
    let stale = dir.join("emitted").join(format!("{:016x}.txt", 1));
    std::fs::write(&stale, format!("{:016x} 000000000000002a\n", made_of(&candidate))).expect("a file");
    assert_eq!(built(&dir, 2).fingerprint(std::slice::from_ref(&candidate)), first, "lowered again, the same code");
    let mut files: Vec<_> = std::fs::read_dir(dir.join("emitted"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    files.sort();
    assert_eq!(files, [format!("{:016x}.txt", 1), format!("{:016x}.txt", 2)]);
}

/// Opening the memo removes the files of builds unused for longer than a
/// build's file is kept, and keeps the recent ones.
#[test]
fn a_build_unused_for_long_loses_its_memo() {
    let dir = Scratch::new("tune-pruned");
    let emitted = dir.join("emitted");
    std::fs::create_dir_all(&emitted).expect("the memo dir");
    let (old, recent) = (emitted.join("old.txt"), emitted.join("recent.txt"));
    for f in [&old, &recent] {
        std::fs::write(f, "").expect("a file");
    }
    let month = std::time::SystemTime::now() - Duration::from_secs(30 * 24 * 3600);
    std::fs::File::options().write(true).open(&old).unwrap().set_modified(month).unwrap();
    built(&dir, 3).fingerprint(&[gemm_candidate(sm86(), super::costs())]);
    assert!(!old.exists() && recent.exists());
}

/// A program is lowered once per tuning: what a fingerprint lowers is kept
/// for the compile and handed over once, the program its target lowers to
/// alone; one the compile lowers first is remembered, so the fingerprint
/// after it finds what it emits without lowering, the same as lowering it.
#[test]
fn a_tuning_lowers_each_program_once() {
    let candidate = gemm_candidate(sm86(), super::costs());
    let [(prog, lowering)] = &candidate[..] else { unreachable!("one program") };
    let alone = crate::lower::standalone(prog.clone(), lowering).expect("lowers").program;
    let (store, keep) = (TuneStore::at(None), tune::Lowerings::default());
    let fingerprinted = store.fingerprint_keeping(std::slice::from_ref(&candidate), Some(&keep));
    let lowered = |store: &TuneStore, keep: &tune::Lowerings| store.lowered(prog, lowering, keep).map(|(_, p)| p);
    assert_eq!(lowered(&store, &keep), Some(alone.clone()), "kept by the fingerprint");
    assert_eq!(lowered(&store, &keep), Some(alone.clone()), "lowered again once taken");
    let (fresh, keep) = (TuneStore::at(None), tune::Lowerings::default());
    assert_eq!(lowered(&fresh, &keep), Some(alone));
    assert_eq!(fresh.fingerprint_keeping(std::slice::from_ref(&candidate), Some(&keep)), fingerprinted);
    assert!(keep.0.lock().unwrap().is_empty(), "nothing lowered: the compile remembered what it emits");
}

/// A program the fingerprint finds does not lower is kept as such, a
/// verdict that stays: its compile is refused without lowering it again,
/// and the bench records the candidate failed.
#[test]
fn a_program_that_does_not_lower_is_lowered_once() {
    let candidate = gemm_candidate(sm86(), super::costs());
    let [(prog, lowering)] = &candidate[..] else { unreachable!("one program") };
    // Half the warps of the grid: the lowering refuses it.
    let prog = &crate::ir::Program { warp_grid: crate::layouts::WarpGrid { rows: 1, cols: 4 }, ..prog.clone() };
    assert!(matches!(crate::lower::standalone(prog.clone(), lowering), Err(crate::lower::Error::Grid { .. })));
    let candidate = vec![(prog.clone(), lowering.clone())];
    let (store, keep) = (TuneStore::at(None), tune::Lowerings::default());
    store.fingerprint_keeping(std::slice::from_ref(&candidate), Some(&keep));
    let m = tune::made_of(prog, lowering);
    assert_eq!(keep.0.lock().unwrap().get(&m), Some(&None), "kept as one that does not lower");
    assert_eq!(store.lowered(prog, lowering, &keep), None);
    assert_eq!(keep.0.lock().unwrap().get(&m), Some(&None), "the verdict stays");
    let mut bench = Bench::default();
    bench.lowered = std::sync::Arc::new(keep);
    assert_eq!(bench.admit(&[0u8], |_| candidate.clone(), &store), vec![tune::Compiled::Failed]);
}

/// A tuner's round gives a candidate the runs 20 ms of device time holds,
/// two to five: a short kernel the five a probe's round gives.
#[test_case(1_000, 5; "a short kernel")]
#[test_case(4_000, 5; "five fit")]
#[test_case(5_000, 4; "four fit")]
#[test_case(8_000, 3; "three fit")]
#[test_case(10_000, 2; "two fit")]
#[test_case(70_000, 2; "a long kernel still runs twice")]
fn a_round_gives_the_runs_its_budget_holds(us: u64, want: usize) {
    assert_eq!(tune::runs(Duration::from_micros(us)), want);
}

/// A stream runs its jobs on the pool at most `width` at once, starting
/// them in the order queued, every result back once; a job's panic resumes
/// on the caller's thread.
#[test]
fn a_stream_runs_width_jobs_at_once_in_order() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (running, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let started = std::sync::Mutex::new(vec![]);
    let run = |job: usize| {
        started.lock().unwrap().push(job);
        let now = running.fetch_add(1, Ordering::SeqCst) + 1;
        most.fetch_max(now, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(5));
        running.fetch_sub(1, Ordering::SeqCst);
        job * 2
    };
    let mut got = tune::streamed(2, run, |stream| {
        (0..12).for_each(|job| stream.push(job));
        std::iter::from_fn(|| stream.next()).collect::<Vec<_>>()
    });
    got.sort_unstable();
    assert_eq!(got, (0..12).map(|j| j * 2).collect::<Vec<_>>());
    assert!(most.load(Ordering::SeqCst) <= 2, "at most two at once");
    let started = started.into_inner().unwrap();
    assert!(started.iter().enumerate().all(|(i, &j)| i.abs_diff(j) < 2), "started in queue order: {started:?}");
    let panicked = std::panic::catch_unwind(|| {
        tune::streamed(
            2,
            |job: usize| assert!(job != 3, "job 3 fails"),
            |stream| {
                (0..6).for_each(|job| stream.push(job));
                while stream.next().is_some() {}
            },
        )
    });
    assert!(panicked.is_err(), "the job's panic reached the caller");
}

/// The rates the vendored floor store prices a floor at: gfx1151's cache
/// ladder and DRAM byte as the Strix Halo's store measured them on
/// 2026-10-27 (fs a streamed byte at each footprint), every field
/// [`Planner::floor`] reads.
const VENDORED_RATES: [(&str, u64); 9] = [
    ("cached_1m", 10542),
    ("cached_2m", 12782),
    ("cached_4m", 21853),
    ("cached_8m", 30412),
    ("cached_16m", 35640),
    ("cached_32m", 38814),
    ("cached_64m", 144723),
    ("cached_128m", 166732),
    ("global_byte", 187054),
];

/// The choice lines of the vendored floor store, synthesized in the store's
/// format-6 grammar: gfx1151's k-nearest tuning of 2026-10-28, the one shape
/// whose readings are known to lie — a 270 ms scores pass whose end stamp
/// landed off the compute, timed 1.3 ms — three candidates under their
/// floor beside the honest sample of the same lines.
const VENDORED_CHOICES: [&str; 2] = [
    "choice|nearest|gfx1151-40sm|Float16|256x1048576x768x64|0000000000000001|0000000000000001|0000000000000001 0:128,128,2,0,16,32,1,~,4,2,1,~,~,~ 1297360 1:128,128,2,1,16,32,1,~,4,2,1,~,~,~ 1305240 2:128,128,1,0,16,32,1,~,4,2,1,~,~,~ 296351560",
    "choice|nearest|gfx1151-40sm|Float16|256x1048576x768x64|0000000000000002|0000000000000001|0000000000000002 1:128,128,2,0,16,32,1,~,4,2,1,~,~,~ 1325040 32:64,32,2,0,16,32,128,~,~,~,~,4,4,1 25261320",
];

/// Every timed candidate of the stores' choice lines clears its physical
/// floor ([`Planner::floor_of`] of its own claim): the least ns a kernel can
/// honestly measure, the bytes it must move at the card's measured rate for
/// them. The store scanned is the vendored one (`VENDORED_RATES` and
/// `VENDORED_CHOICES`, written to a scratch store as `probe::stored` reads
/// it), so the check runs anywhere and fails the day the floor stops
/// flagging the three lied readings; `TK_FLOOR_STORES` (`:`-separated) adds
/// real stores on machines that hold them, where nothing under its floor may
/// be anything but the same lie — an honest candidate never refused.
#[test]
fn timed_candidates_clear_their_physical_floor() {
    fn dtype_of(name: &str) -> Option<ScalarDType> {
        match name {
            "Float16" => Some(ScalarDType::Float16),
            "BFloat16" => Some(ScalarDType::BFloat16),
            "Int8" => Some(ScalarDType::Int8),
            _ => None,
        }
    }
    fn target(device: &str) -> Option<crate::atoms::Target> {
        use svod_dtype::{AmdArch, CudaArch, GpuArch};
        let (arch, sms) = device.rsplit_once('-')?;
        let sms = sms.strip_suffix("sm")?.parse().ok()?;
        let arch = match AmdArch::parse(arch) {
            Some(a) => GpuArch::Amd(a),
            None => GpuArch::Cuda(arch.trim_end_matches('a').parse::<CudaArch>().ok()?),
        };
        let xcds = crate::atoms::CARDS.iter().find(|c| c.1 == arch).map_or(
            match arch {
                GpuArch::Amd(a) if a.is_cdna() => 8,
                _ => 1,
            },
            |c| c.3,
        );
        Some(crate::atoms::Target::offline(arch, sms, xcds))
    }
    fn floor(
        planner: &crate::ops::config::Planner,
        op: &str,
        dtype: ScalarDType,
        shape: &[usize],
        code: &str,
    ) -> Option<f64> {
        use crate::kernels::nearest::{Floors, Metric};
        use crate::ops::config::{AttentionProblem, ConvProblem, GemmProblem, NearestProblem, Tiled};
        match op {
            "nearest" => {
                let (n, m, d, k) = match shape[..] {
                    [n, m, d] => (n, m, d, 1),
                    [n, m, d, k] => (n, m, d, k),
                    _ => return None,
                };
                let p = NearestProblem {
                    n,
                    m,
                    d,
                    dtype,
                    metric: Metric::SquaredL2,
                    floors: Floors::default(),
                    groups: false,
                    k,
                };
                planner.floor_of(&p, &Tiled::of_code(code)?)
            }
            "gemm" | "linear" => {
                let &[cap, m, n, k] = shape else { return None };
                let p = GemmProblem { batches: cap.max(1), m, n, k, halves: 1, rope: None };
                planner.floor_of(&p, &Tiled::of_code(code)?)
            }
            "attention" => {
                let &[cap, t, tk, heads, _kv, d] = shape else { return None };
                let p = AttentionProblem { tiles: cap * heads, t, tk, d, causal: false, bias: false };
                planner.floor_of(&p, &Tiled::of_code(code)?)
            }
            "conv2d" => {
                let (g, images) = match shape[..] {
                    [cin, cout, h, w, stride] => (
                        crate::kernels::conv::ConvGeom {
                            h,
                            w,
                            cin,
                            cout,
                            kernel: [3, 3],
                            stride: [stride, stride],
                            pad: [1, 1],
                            dilation: [1, 1],
                        },
                        1,
                    ),
                    [images, h, w, cin, cout, k0, k1, s0, s1, p0, p1, d0, d1] => (
                        crate::kernels::conv::ConvGeom {
                            h,
                            w,
                            cin,
                            cout,
                            kernel: [k0, k1],
                            stride: [s0, s1],
                            pad: [p0, p1],
                            dilation: [d0, d1],
                        },
                        images,
                    ),
                    _ => return None,
                };
                let [ho, wo] = g.out_hw();
                let p = ConvProblem {
                    gemm: GemmProblem { batches: 1, m: images * ho * wo, n: g.cout, k: g.k(), halves: 1, rope: None },
                    geom: g,
                };
                let c: crate::kernels::conv::ConvCfg = Tiled::of_code(code)?;
                p.claim(planner, &c.gemm).ok().map(|claim| planner.floor(&claim))
            }
            _ => None,
        }
    }

    /// Every timed candidate of the choice lines under `dir`, as its ns over
    /// its floor, fastest liars first.
    fn scan(dir: &std::path::Path) -> Vec<(f64, String)> {
        let mut worst: Vec<(f64, String)> = vec![];
        let Ok(files) = std::fs::read_dir(dir) else { return worst };
        let mut planners: std::collections::HashMap<String, Option<crate::ops::config::Planner>> = Default::default();
        for file in files.flatten() {
            let text = std::fs::read_to_string(file.path()).unwrap_or_default();
            let name = file.path().file_stem().unwrap().to_string_lossy().to_string();
            for line in text.lines() {
                let fields: Vec<&str> = line.split('|').collect();
                if fields.first() != Some(&"choice") || fields.len() != 8 {
                    continue;
                }
                let (op, dtype, shape) = (fields[1], fields[3], fields[4]);
                let Some(dtype) = dtype_of(dtype) else { continue };
                let Ok(shape) = shape.split('x').map(|n| n.parse::<usize>()).collect::<Result<Vec<_>, _>>() else {
                    continue;
                };
                let planner = planners.entry(fields[2].to_string()).or_insert_with(|| {
                    let t = target(fields[2])?;
                    crate::probe::stored(dir, &t).and_then(|m| crate::ops::config::Planner::measured(t, m))
                });
                let Some(planner) = planner else { continue };
                let Some(pairs) = fields[7].split_once(' ').map(|(_, p)| p) else { continue };
                for pair in pairs.split(' ').collect::<Vec<_>>().chunks(2) {
                    let [f, ns] = pair else { continue };
                    let Some(code) = f.split_once(':').map(|(_, c)| c) else { continue };
                    let Some(ns) = ns.parse::<u64>().ok() else { continue };
                    let Some(f) = floor(planner, op, dtype, &shape, code) else { continue };
                    if f > 0.0 {
                        worst.push((ns as f64 / f, format!("{name} {op} {shape:?} {code}")));
                    }
                }
            }
        }
        worst.sort_by(|a, b| a.0.total_cmp(&b.0));
        worst
    }
    let lie = |what: &str| what.contains("gfx1151") && what.contains("nearest [256, 1048576, 768, 64] 128,128,2,");

    let dir = Scratch::new("floor-store");
    let file = TuneStore::at(Some(dir.to_path_buf())).file("gfx1151-40sm").expect("a rooted store");
    let lines: Vec<String> = VENDORED_RATES
        .iter()
        .map(|&(op, fs)| {
            format!("value|probe:{op}|gfx1151-40sm|Int32||00000000000000c0|00000000000000aa|00000000000000bb {fs}")
        })
        .chain(VENDORED_CHOICES.iter().map(|line| (*line).to_string()))
        .collect();
    std::fs::write(file, format!("format {FORMAT}\n{}\n", lines.join("\n"))).expect("a store file");

    let vendored = scan(&dir);
    println!("vendored candidates: {}", vendored.len());
    for (ratio, what) in vendored.iter().take(8) {
        println!("{ratio:8.3} {what}");
    }
    assert!(vendored.len() > 3, "the vendored store carries its honest sample: {vendored:?}");
    let under: Vec<&String> = vendored.iter().filter(|&&(ratio, _)| ratio < 1.0).map(|(_, what)| what).collect();
    assert_eq!(
        under.len(),
        3,
        "under the floor: {under:?}\nthe least above it: {:?}",
        vendored.iter().find(|&&(ratio, _)| ratio >= 1.0)
    );
    for what in under {
        assert!(lie(what), "an honest candidate refused: {what}");
    }

    if let Ok(list) = std::env::var("TK_FLOOR_STORES") {
        for store in list.split(':').filter(|s| !s.is_empty()) {
            let worst = scan(std::path::Path::new(store));
            println!("-- {store}: {} candidates", worst.len());
            for (ratio, what) in worst.iter().take(8) {
                println!("{ratio:8.3} {what}");
            }
            for (_, what) in worst.iter().filter(|&&(ratio, _)| ratio < 1.0) {
                assert!(lie(what), "an honest candidate refused: {what}");
            }
        }
    }
}

/// A candidate measuring under its floor never becomes the pick — not
/// measured, not read back from a store that holds the number: the choice is
/// the next candidate's. One that refuses everything leaves the order as it
/// ran: a floor that refuses every candidate is itself what is wrong.
#[test]
fn an_underfloor_measurement_is_never_the_pick() {
    let dir = Scratch::new("tune-floor");
    let k = key(&[128, 128, 128], &[0, 1]);
    let cfgs = [gemm_cfg([64, 64, 32], 2, [2, 2], true), gemm_cfg([128, 64, 32], 2, [2, 2], true)];
    let programs = || fingerprint(&[gemm_programs(cfgs[0]), gemm_programs(cfgs[1])]);
    TuneStore::at(Some(dir.to_path_buf())).select(&k, 2, programs, || vec![Some(1), Some(4)]);
    let build = |c: crate::kernels::gemm::GemmCfg| gemm_programs(c);
    let floor = |c: &crate::kernels::gemm::GemmCfg| (c == &cfgs[0]).then_some(3.0);
    let chosen = TuneStore::at(Some(dir.to_path_buf())).choose(
        &k,
        &cfgs,
        build,
        &mut Bench::default(),
        Pass::Whole,
        Some(&floor),
    );
    assert_eq!(chosen.map(|timed| timed[0].0), Some(1), "the under-floor candidate is refused");
    let every = |_: &crate::kernels::gemm::GemmCfg| Some(9.0);
    let kept = TuneStore::at(Some(dir.to_path_buf())).choose(
        &k,
        &cfgs,
        build,
        &mut Bench::default(),
        Pass::Whole,
        Some(&every),
    );
    assert_eq!(kept.map(|timed| timed[0].0), Some(0), "a floor refusing everything is ignored");
}
