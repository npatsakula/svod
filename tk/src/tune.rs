//! Measured config choice per shape, made at graph build: an op's ranked
//! configs are admitted ([`Bench`]) and, on a store miss, timed; the winner
//! is kept in the process memo and on disk.
//!
//! The store is one line per [`Entry`] (`kind|key candidates`) under a
//! `format N` line ([`FORMAT`]) in `$SVOD_TK_TUNE_DIR` (else
//! `$XDG_CACHE_HOME/svod/tk_tune`, else `$HOME/.cache/svod/tk_tune`), one
//! file per device and crate version. A line's key carries a digest of the
//! device's compiler ([`compiler`]) and a fingerprint of the kernels the
//! candidates emit ([`fingerprint`]), so a toolchain or kernel change
//! re-measures and nothing else does; its candidates carry each config's
//! own code beside its index, so a line reads without re-deriving the list
//! its digest names ([`crate::rank`]). An unreadable or unwritable store is
//! a miss, never an error; a line the store cannot read is dropped at its
//! next write, and a file of another format whole (a format bump orphans
//! every file: the next run re-probes and re-tunes every shape).
//! `SVOD_TK_TUNE=0`, or [`scoped`] on a thread, turns candidate timing off:
//! the first ranked config that compiles without spilling is used (that
//! verdict is still stored), while a device's probes still measure.

use std::borrow::BorrowMut;
use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use svod_dtype::default_device::{default_device, with_default_device};
use svod_dtype::{DType, DeviceSpec, ScalarDType};
use svod_ir::UOp;
use svod_runtime::benchmark::{CLOCK_WARMUP, round_robin_min, warm_clock};
use svod_runtime::{ExecutionPlan, KernelResources};
use svod_tensor::{PrepareConfig, Tensor};

use crate::atoms::Target;
use crate::ir::{Param, Program};
use crate::launch;
use crate::lower::Lowering;

/// The store's line format, bumped when a line's meaning changes; a file of
/// another format, or without the line, is dropped whole. 6 carries each
/// candidate's config code beside its index (`idx:code ns` a timed pair,
/// `idx:code` an admitted one), where 5 held indices alone into the list the
/// line's digest names — a ranking change used to orphan every choice line,
/// and now the line's own candidates replay ([`crate::rank`]). 5
/// fingerprinted the emitted kernels, where 4 fingerprinted the programs
/// and lowerings, targets and costs whole.
pub(crate) const FORMAT: u32 = 6;
/// Measurement: the rounds a contender is timed (the first over every
/// candidate, the rest over the contenders), and the rounds the two leaders
/// then run again ([`rounds`]).
const ROUNDS: usize = 4;
/// How far a timing goes ([`rounds`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pass {
    /// Every round, then the leaders' rounds: the order is the choice.
    Whole,
    /// The first round and one more of its contenders: the order only
    /// chooses what a later pass times again with the rest of its kind (a
    /// tuning's leading tiles, whose pipelines the twins stage times with
    /// them), and the rounds that would settle its leaders run there.
    Screen,
}
/// Search bound: candidates from which the first round cuts the field to
/// its [`CONTENDER`]s; fewer run every round, a cut saving little and a
/// short list being what a decode step's splits or a coordinate descent
/// ask, where the candidates sit within a few percent of each other. A
/// list of 8 to 15 is a shortlist all the same (gfx1151 admits 15 tiles of
/// a split nearest at depth 768, a third of them past 1.4 times the best).
const CUT_FROM: usize = 8;
/// Search bound: rounds past the first go to the candidates within this
/// factor of the first round's best; the rest cannot win (a round's best of [`RUNS`] runs
/// after a sustain sits within a few percent of the candidate's settled
/// time), and a shortlist is mostly losers. The twins stage keeps the tiles
/// within it of the leader for the same reason.
pub(crate) const CONTENDER: f64 = 1.2;
/// Search bound: rounds past the second go to the contenders within this
/// factor of the best so far. Two rounds in, a candidate's best sits within
/// a few percent of its settled time, and a flat field (a split nearest's
/// tiles, the GEMM's twins) keeps most of its contenders through every
/// round of a 1.2 band: replaying the rounds recorded on sm_86, gfx1201 and
/// gfx1151 (nearest at depths 512–1024, the 4096³ GEMM, an attention; 27
/// timed lists) under a cut at 1.02–1.08 moved no pick, and one at 1.05
/// times a fourth fewer candidates.
const SETTLED: f64 = 1.05;
/// Search bound: a candidate whose first run takes this factor of the best
/// time so far is timed no further, and no later round of a short list
/// runs it: it cannot win. The first timed run follows an untimed one, and
/// over 908 candidates on sm_86 (nearest, the 4096³ GEMM) it sat within
/// 0.93–1.15 of the candidate's settled time: one aborted settles past 1.7
/// times the best.
pub(crate) const HOPELESS: f64 = 2.0;
/// Measurement: profiled runs per candidate per round, the most a tuner's
/// round gives one ([`runs`]) and what the leaders' rounds and the probes give.
const RUNS: usize = 5;
/// Measurement: the device time a tuner's round gives a candidate's profiled
/// runs, at least [`MIN_RUNS`] of them ([`runs`]): a 10 ms kernel's best of
/// two sits where its best of five does, within a round's drift.
const RUN_BUDGET: Duration = Duration::from_millis(20);
/// Measurement: the fewest profiled runs a tuner's round gives a candidate.
const MIN_RUNS: usize = 2;
/// Measurement: the most profiled runs a candidate's minimum takes to prove
/// consistent: a run under [`STEADY`] of the minimum so far resets the
/// count, three over it settle the candidate. A clock still lifting shows a
/// falling minimum run to run (a 3060 read a nearest search at 4.8 ms
/// against its warm 0.87 while its every timed run rode the sagged clock);
/// a warm kernel's minimum is flat within a run's drift.
const STABLE_RUNS: usize = 3;
/// Measurement: a run within this of the minimum so far counts stable.
const STEADY: f64 = 0.97;
/// Measurement: the most profiled runs a candidate is given to settle.
const MOST_RUNS: usize = 24;

/// The profiled runs a tuner's round gives a candidate whose last run took
/// `t`: those [`RUN_BUDGET`] holds, [`MIN_RUNS`] to [`RUNS`]. A candidate
/// of [`SUSTAIN`] or more is not sustained: its runs keep the clock up, and
/// its first timed run counts among them.
pub(crate) fn runs(t: Duration) -> usize {
    (RUN_BUDGET.as_nanos().div_ceil(t.as_nanos().max(1)) as usize).clamp(MIN_RUNS, RUNS)
}
/// Measurement: device time of back-to-back runs ahead of each candidate's
/// profiled runs, keeping the lifted clock up where their host overhead would let a short
/// kernel's clock sag.
const SUSTAIN: Duration = Duration::from_millis(5);
/// Measurement: the most launches a [`SUSTAIN`] queues, however short the kernel.
const SUSTAIN_LAUNCHES: u128 = 1024;

fn digest(value: &impl Hash) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// What a choice is keyed by. Every field is cheap, so a warm memo answers
/// without building a program; `candidates` digests the configs and whatever
/// else the programs vary with that `shape` does not spell out (an epilogue).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TuneKey {
    pub op: &'static str,
    /// Arch and SM count, e.g. `sm_86-28sm`.
    pub device: String,
    pub dtype: ScalarDType,
    pub shape: Vec<usize>,
    pub candidates: u64,
    /// The device's [`compiler`] digest.
    pub compiler: u64,
}

impl TuneKey {
    pub(crate) fn new(
        op: &'static str,
        target: &Target,
        compiler: u64,
        dtype: ScalarDType,
        shape: &[usize],
        candidates: &impl Hash,
    ) -> Self {
        Self { op, device: device(target), dtype, shape: shape.to_vec(), candidates: digest(candidates), compiler }
    }

    /// The store line of a `kind` of entry for the candidate programs
    /// fingerprinted as `programs`.
    pub(crate) fn line(&self, kind: Kind, programs: u64) -> String {
        format!("{}|{:016x}|{programs:016x}", self.candidates(kind), self.compiler)
    }

    /// A line's key up to the candidates' digest: what a [`TuneStore::replay`]
    /// matches, whatever compiler and programs keyed the line.
    pub(crate) fn candidates(&self, kind: Kind) -> String {
        format!("{}|{:016x}", self.shape(kind), self.candidates)
    }

    /// A line's key up to the shape: the lines of one shape, any candidates.
    fn shape(&self, kind: Kind) -> String {
        let shape: Vec<String> = self.shape.iter().map(usize::to_string).collect();
        format!("{}|{}|{}|{:?}|{}", kind.name(), self.op, self.device, self.dtype, shape.join("x"))
    }
}

/// A target's name in the store: arch and SM count ([`TuneKey::device`]).
pub(crate) fn device(target: &Target) -> String {
    format!("{}-{}sm", target.arch.target_name(), target.sms.unwrap_or(0))
}

/// A digest of `spec`'s compiler identity (backend, toolchain, flags) for
/// [`TuneKey::compiler`], once per device; 0 where no device opens.
pub(crate) fn compiler(spec: &DeviceSpec) -> u64 {
    static DIGESTS: LazyLock<Once<DeviceSpec, u64>> = LazyLock::new(Mutex::default);
    once(&DIGESTS, spec, || {
        let device = svod_runtime::DEVICE_FACTORIES.device(spec, svod_device::registry::registry());
        device.map_or(0, |device| digest(&device.compiler.cache_key()))
    })
}

/// A candidate's programs, launched in order (a kernel and the merge of
/// its partial results, say).
pub type Candidate = Vec<(Program, Lowering)>;

/// The fingerprint of candidates' programs: the kernels they emit
/// ([`emitted`]), so a field of the target or the costs no instruction
/// reads keys nothing, added, renamed or deleted. What a program emits is
/// remembered by what it is made of, by the process and, for later
/// processes of the same build, by the global store
/// ([`TuneStore::fingerprint`]): lowering a shortlist takes 0.05–1.6 s,
/// and a warm process lowers nothing.
pub(crate) fn fingerprint(candidates: &[Candidate]) -> u64 {
    TuneStore::global().fingerprint(candidates)
}

/// What `prog` emits under `lowering`: the structural hash of the program
/// it lowers to (its instructions, grid and parameters) and the text the
/// backend's compiler is handed, which a renderer change moves. Lowered
/// outside any origin scope, which would key the call site; `None` where it
/// does not lower or render, and the caller keys it by its inputs: a
/// program that fails to lower fails the same way until they change.
fn emitted(prog: &Program, lowering: &Lowering) -> Option<(u64, Arc<UOp>)> {
    let _capture = svod_ir::origin::capture_for_thread(false);
    let _scope = svod_ir::origin::install(None);
    let lowered = crate::lower::standalone(prog.clone(), lowering)
        .inspect_err(|e| tracing::debug!(target: crate::ops::PLAN, name = prog.name, error = %e, "does not lower"))
        .ok()?;
    let emitted = digest(&(digest(&lowered.program), lowered.text(&lowering.device.target).ok()?));
    Some((emitted, lowered.program))
}

/// A program lowered (or found not to lower), with what it emits.
type Lowered = HashMap<u64, Option<(u64, Arc<UOp>)>>;

/// What a program is made of, which keys what it emits: the program and
/// its lowering whole.
pub(crate) fn made_of(prog: &Program, lowering: &Lowering) -> u64 {
    digest(&format!("{prog:?}{lowering:?}"))
}

/// Programs lowered for a fingerprint, by what they are made of, kept for
/// the compile that follows: a tuning lowers each program once, with the
/// digest of what it emits ([`emitted_key`] of the candidate it belongs to).
/// One that did not lower is kept as `None`, which stays: it fails the same
/// way until its inputs change, so its compile is refused without lowering it
/// again.
#[derive(Default)]
pub(crate) struct Lowerings(pub(crate) Mutex<Lowered>);

impl Lowerings {
    /// What the fingerprint left for `made_of`: what the program emits and
    /// the program, handed over once; `Some(None)` where it did not lower;
    /// `None` where nothing was.
    fn take(&self, made_of: u64) -> Option<Option<(u64, Arc<UOp>)>> {
        let mut kept = self.0.lock().expect("lowerings");
        match kept.get(&made_of) {
            Some(None) => Some(None),
            Some(Some(_)) => kept.remove(&made_of),
            None => None,
        }
    }
}

/// Values per key, each made once: a caller asking for a key in flight
/// waits for its value rather than making another.
type Once<K, V> = Mutex<HashMap<K, Arc<OnceLock<V>>>>;

fn once<K: Clone + Eq + Hash, V: Clone>(map: &Once<K, V>, key: &K, init: impl FnOnce() -> V) -> V {
    let cell = map.lock().expect("tune memo").entry(key.clone()).or_default().clone();
    cell.get_or_init(init).clone()
}

/// What a store line records, under its kind, key and fingerprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entry {
    /// The candidates that ran, fastest first, each with its index into the
    /// list the line's key digests, its config's code and its device ns.
    Choice(Arc<[(usize, String, u64)]>),
    /// A measured number: a probe's cost.
    Value(u64),
    /// The ranked configs admission kept (each with its index and code), or
    /// the one its search for a first clean config found.
    Admitted(Arc<[(usize, String)]>),
}

/// What a [`TuneStore::replay`] found for one key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lookup {
    pub kind: Kind,
    /// The key up to its candidates' digest ([`TuneKey::candidates`]).
    pub key: String,
    /// The entry read, `None` when missing.
    pub entry: Option<Entry>,
    pub found: Found,
}

/// Which of a key's lines a replay read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Found {
    /// The line of the programs this build emits.
    Own,
    /// No line of this build's programs; `lines` of other builds (another
    /// kernel text), all holding the one entry read.
    Built { lines: usize },
    /// As [`Found::Built`], but the lines disagree: the first in the file's
    /// order is read.
    Disagree { lines: usize },
    /// No line of the key; `others` lines of the kind for the same op,
    /// device, dtype and shape under other candidate lists (another ranking,
    /// another salt or rule).
    Missing { others: usize },
}

/// An [`Entry`]'s kind, leading its line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Choice,
    Value,
    Admitted,
    /// An admission of one: the first clean config of a ranked list.
    First,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Choice => "choice",
            Kind::Value => "value",
            Kind::Admitted => "admitted",
            Kind::First => "first",
        }
    }
}

impl Entry {
    /// The entry's candidates as its line holds them after the key: a timed
    /// pair its index, code and ns; an admitted one its index and code.
    pub(crate) fn render(&self) -> String {
        match self {
            Entry::Choice(timed) => {
                timed.iter().map(|(i, c, ns)| format!("{i}:{c} {ns}")).collect::<Vec<_>>().join(" ")
            }
            Entry::Value(v) => v.to_string(),
            Entry::Admitted(at) => at.iter().map(|(i, c)| format!("{i}:{c}")).collect::<Vec<_>>().join(" "),
        }
    }

    /// The entry a line of `kind` with `fields` after its key records.
    pub(crate) fn parse(kind: &str, fields: &[&str]) -> Option<Self> {
        let at = |f: &str| -> Option<(usize, String)> {
            let (i, code) = f.split_once(':')?;
            Some((i.parse().ok()?, code.to_string()))
        };
        match (kind, fields) {
            ("choice", pairs) if !pairs.is_empty() && pairs.len().is_multiple_of(2) => Some(Entry::Choice(
                pairs
                    .chunks(2)
                    .map(|p| {
                        let (i, code) = at(p[0])?;
                        Some((i, code, p[1].parse().ok()?))
                    })
                    .collect::<Option<_>>()?,
            )),
            ("value", &[v]) => Some(Entry::Value(v.parse().ok()?)),
            ("admitted", ats) if !ats.is_empty() => {
                Some(Entry::Admitted(ats.iter().map(|f| at(f)).collect::<Option<_>>()?))
            }
            ("first", &[one]) => Some(Entry::Admitted([at(one)?].into())),
            _ => None,
        }
    }
}

/// Removes the files under `dir` not modified for [`EMITTED_KEPT`].
fn prune(dir: &std::path::Path) {
    let _ = std::fs::create_dir_all(dir);
    let Ok(files) = std::fs::read_dir(dir) else { return };
    for file in files.flatten() {
        let age = file.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok());
        if age.is_some_and(|age| age > EMITTED_KEPT) {
            let _ = std::fs::remove_file(file.path());
        }
    }
}

/// A memo of this process's entries, backed by one file per device under `root`.
#[derive(Debug)]
pub struct TuneStore {
    root: Option<PathBuf>,
    /// What a [`Self::replay`] found per key asked, in order; `None` for a
    /// store that measures and writes.
    replayed: Option<Mutex<Vec<Lookup>>>,
    /// The entry per kind and key, `None` where nothing could be measured.
    memo: Once<(Kind, TuneKey), Option<Entry>>,
    /// What programs emit by what they are made of ([`Self::fingerprint`]).
    emitted: OnceLock<Mutex<HashMap<u64, u64>>>,
    /// The running executable's [`build`], which names its file of what
    /// programs emit; `None`: the memo stays in memory.
    pub(crate) build: Option<u64>,
}

/// Search bound: how long a build's file of what programs emit
/// ([`TuneStore::fingerprint`]) outlives its last use: a build not run for
/// as long is gone, and another lowers its programs once again.
const EMITTED_KEPT: Duration = Duration::from_secs(7 * 24 * 3600);

/// The running executable's identity: a digest of its path, length and
/// modification time, which every rebuild moves (its contents are not read).
fn build() -> Option<u64> {
    let exe = std::env::current_exe().ok()?;
    let meta = std::fs::metadata(&exe).ok()?;
    Some(digest(&(exe, meta.len(), meta.modified().ok()?)))
}

impl TuneStore {
    /// The store rooted at `root` (`None`: memory only).
    pub fn at(root: Option<PathBuf>) -> Self {
        Self { root, replayed: None, memo: Mutex::default(), emitted: OnceLock::new(), build: build() }
    }

    /// The store at `root` read back without a device, written to never
    /// (its `emitted` memo included): the entries a tuner reads are the
    /// lines of each key up to its candidates' digest ([`TuneKey::candidates`]),
    /// whatever compiler keyed them (no device opens here to name its own)
    /// and whatever programs: the line of the programs this build emits
    /// where there is one, else the other builds' lines. Nothing is
    /// measured: a key without a line answers `None`, as an unmeasured one
    /// does. [`Self::lookups`] says what each key found.
    pub fn replay(root: PathBuf) -> Self {
        Self { replayed: Some(Mutex::default()), ..Self::at(Some(root)) }
    }

    /// Whether this is a [`Self::replay`].
    pub(crate) fn replays(&self) -> bool {
        self.replayed.is_some()
    }

    /// The directory this store keeps its device files under, `None` for a
    /// memory-only one: where a card's probes and oracle verdicts live.
    pub fn root(&self) -> Option<&std::path::Path> {
        self.root.as_deref()
    }

    /// What a [`Self::replay`] found for each key asked, in order.
    pub fn lookups(&self) -> Vec<Lookup> {
        self.replayed.as_ref().map(|r| r.lock().expect("replay log").clone()).unwrap_or_default()
    }

    /// The entry a replay reads for `key` ([`Self::replay`]), logged in
    /// [`Self::lookups`].
    fn replayed(&self, kind: Kind, key: &TuneKey, programs: u64, valid: impl Fn(&Entry) -> bool) -> Option<Entry> {
        let entries = self.entries(key);
        let (prefix, shape) = (format!("{}|", key.candidates(kind)), format!("{}|", key.shape(kind)));
        let mut lines: Vec<(&String, &Entry)> =
            entries.iter().filter(|(line, entry)| line.starts_with(&prefix) && valid(entry)).collect();
        lines.sort_by_key(|(line, _)| line.as_str());
        let own = lines.iter().find(|(line, _)| line.ends_with(&format!("|{programs:016x}")));
        let (found, entry) = match (own, lines.first()) {
            (Some(&(_, entry)), _) => (Found::Own, Some(entry)),
            (None, Some(&(_, entry))) if lines.iter().all(|(_, e)| *e == entry) => {
                (Found::Built { lines: lines.len() }, Some(entry))
            }
            (None, Some(&(_, entry))) => (Found::Disagree { lines: lines.len() }, Some(entry)),
            (None, None) => {
                let others = entries.keys().filter(|line| line.starts_with(&shape)).count();
                (Found::Missing { others }, None)
            }
        };
        let lookup = Lookup { kind, key: key.candidates(kind), entry: entry.cloned(), found };
        tracing::debug!(target: crate::ops::PLAN, ?lookup, "replayed");
        self.replayed.as_ref().expect("a replay").lock().expect("replay log").push(lookup);
        entry.cloned()
    }

    /// [`fingerprint`] under this store: what each program emits is read
    /// from what the process and its build's file (`emitted/<build>.txt`
    /// under the root) know by the digest of what it is made of (its program
    /// and lowering whole), else lowered, in parallel, and added to both. A
    /// program that does not lower is keyed by what it is made of. The memo
    /// keys inputs, so it is the build's own: a rebuild, whose lowering or
    /// renderer may emit otherwise for the same inputs, lowers every program
    /// once and finds the store's entries again wherever the code is the
    /// same. Opening the file marks it used and removes the builds' files
    /// unused for [`EMITTED_KEPT`]; each line is appended by one write, so
    /// processes of one build share the file.
    pub(crate) fn fingerprint(&self, candidates: &[Candidate]) -> u64 {
        self.fingerprint_keeping(candidates, None).0
    }

    /// [`Self::fingerprint`], the programs it lowers kept in `keep` for
    /// their compile, and the digest of what each candidate's programs emit:
    /// candidates sharing one emit the same kernel ([`emitted_key`]).
    pub(crate) fn fingerprint_keeping(&self, candidates: &[Candidate], keep: Option<&Lowerings>) -> (u64, Vec<u64>) {
        use rayon::prelude::*;
        let made_of: Vec<Vec<u64>> =
            candidates.iter().map(|c| c.iter().map(|(p, l)| made_of(p, l)).collect()).collect();
        let memo = self.emitted_memo();
        let missing: Vec<(u64, &(Program, Lowering))> = {
            let known = memo.lock().expect("emitted memo");
            let mut seen = std::collections::HashSet::new();
            let programs = made_of.iter().flatten().zip(candidates.iter().flatten());
            programs.filter(|(m, _)| !known.contains_key(*m) && seen.insert(**m)).map(|(&m, p)| (m, p)).collect()
        };
        let found: Vec<(u64, u64)> = missing
            .into_par_iter()
            .map(|(m, (prog, lowering))| {
                let lowered = emitted(prog, lowering);
                if let Some(keep) = keep {
                    keep.0
                        .lock()
                        .expect("lowerings")
                        .insert(m, lowered.as_ref().map(|(emitted, program)| (*emitted, program.clone())));
                }
                lowered.map_or((m, m), |(emitted, _)| (m, emitted))
            })
            .collect();
        self.record_emitted(&found);
        let known = memo.lock().expect("emitted memo");
        let emitted: Vec<Vec<u64>> = made_of.iter().map(|c| c.iter().map(|m| known[m]).collect()).collect();
        (digest(&emitted), emitted.iter().map(digest).collect())
    }

    /// `prog` lowered under `lowering` ([`crate::lower::standalone`]),
    /// without origins, from `keep` where a fingerprint left it (or found
    /// it does not lower), else lowered here and what it emits remembered
    /// as a fingerprint would; `None` where it does not lower. What the
    /// program emits comes with it.
    pub(crate) fn lowered(&self, prog: &Program, lowering: &Lowering, keep: &Lowerings) -> Option<(u64, Arc<UOp>)> {
        let m = made_of(prog, lowering);
        if let Some(kept) = keep.take(m) {
            return kept;
        }
        let (emitted, program) = emitted(prog, lowering)?;
        if !self.emitted_memo().lock().expect("emitted memo").contains_key(&m) {
            self.record_emitted(&[(m, emitted)]);
        }
        Some((emitted, program))
    }

    /// The digest of what `programs` emit, where a fingerprint or a compile
    /// lowered them (else `None`): what a candidate is keyed by, its twins
    /// (the same emitted kernel) sharing it.
    pub(crate) fn emitted_key(&self, programs: &Candidate) -> Option<u64> {
        let memo = self.emitted_memo().lock().expect("emitted memo");
        let emitted: Vec<u64> =
            programs.iter().map(|(p, l)| memo.get(&made_of(p, l)).copied()).collect::<Option<_>>()?;
        Some(digest(&emitted))
    }

    /// What programs emit by what they are made of: this process's and its
    /// build's file's.
    fn emitted_memo(&self) -> &Mutex<HashMap<u64, u64>> {
        self.emitted.get_or_init(|| {
            if let Some(dir) = self.emitted_dir() {
                prune(&dir);
            }
            let file = self.emitted_file();
            let open = |f: &PathBuf| std::fs::OpenOptions::new().create(true).append(true).open(f);
            if let Some(Ok(f)) = file.as_ref().map(open) {
                let _ = f.set_modified(std::time::SystemTime::now());
            }
            let text = file.as_ref().and_then(|f| std::fs::read_to_string(f).ok()).unwrap_or_default();
            let hex = |s: &str| u64::from_str_radix(s, 16).ok();
            Mutex::new(
                text.lines().filter_map(|l| l.split_once(' ').and_then(|(a, b)| Some((hex(a)?, hex(b)?)))).collect(),
            )
        })
    }

    fn emitted_dir(&self) -> Option<PathBuf> {
        self.root.as_ref().filter(|_| !self.replays()).map(|root| root.join("emitted"))
    }

    fn emitted_file(&self) -> Option<PathBuf> {
        self.emitted_dir().zip(self.build).map(|(dir, build)| dir.join(format!("{build:016x}.txt")))
    }

    /// Adds `(made of, emitted)` pairs to the memo and the build's file.
    fn record_emitted(&self, found: &[(u64, u64)]) {
        if let (Some(file), false) = (self.emitted_file(), found.is_empty()) {
            let append = std::fs::OpenOptions::new().create(true).append(true).open(file);
            if let Ok(mut f) = append {
                for (m, e) in found {
                    let _ = std::io::Write::write_all(&mut f, format!("{m:016x} {e:016x}\n").as_bytes());
                }
            }
        }
        self.emitted_memo().lock().expect("emitted memo").extend(found.iter().copied());
    }

    /// The process-wide store per the environment (see the module docs).
    pub fn global() -> &'static Self {
        static STORE: OnceLock<TuneStore> = OnceLock::new();
        STORE.get_or_init(|| {
            let var = |name| std::env::var_os(name).map(PathBuf::from);
            let root = var("SVOD_TK_TUNE_DIR")
                .or_else(|| var("XDG_CACHE_HOME").map(|c| c.join("svod/tk_tune")))
                .or_else(|| var("HOME").map(|h| h.join(".cache/svod/tk_tune")));
            Self::at(root.filter(|dir| std::fs::create_dir_all(dir).is_ok()))
        })
    }

    fn path(&self, key: &TuneKey) -> Option<PathBuf> {
        self.file(&key.device)
    }

    /// The file of a [`device`]'s entries.
    pub(crate) fn file(&self, device: &str) -> Option<PathBuf> {
        let name: String = device.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        self.root.as_ref().map(|root| root.join(format!("{name}-v{}.txt", env!("CARGO_PKG_VERSION"))))
    }

    /// Every readable `line -> entry` of the key's device file; empty when
    /// unreadable or of another format.
    pub(crate) fn entries(&self, key: &TuneKey) -> HashMap<String, Entry> {
        let Some(text) = self.path(key).and_then(|p| std::fs::read_to_string(p).ok()) else { return HashMap::new() };
        if text.lines().next() != Some(format!("format {FORMAT}").as_str()) {
            return HashMap::new();
        }
        text.lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let line = fields.next()?;
                let numbers: Vec<&str> = fields.collect();
                Some((line.to_string(), Entry::parse(line.split('|').next()?, &numbers)?))
            })
            .collect()
    }

    /// Re-read, merge and replace the file atomically, one writer of a file
    /// at a time in this process, so writers of other processes lose at
    /// most each other's newest line, never the file, and this process's
    /// lose nothing.
    fn put(&self, key: &TuneKey, line: String, entry: &Entry) {
        static WRITERS: LazyLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = LazyLock::new(Mutex::default);
        static TEMPS: AtomicU64 = AtomicU64::new(0);
        let Some(path) = self.path(key).filter(|_| !self.replays()) else { return };
        let writer = WRITERS.lock().unwrap_or_else(PoisonError::into_inner).entry(path.clone()).or_default().clone();
        let _writing = writer.lock().unwrap_or_else(PoisonError::into_inner);
        let mut entries = self.entries(key);
        entries.insert(line, entry.clone());
        let mut lines: Vec<String> = entries.iter().map(|(line, entry)| format!("{line} {}", entry.render())).collect();
        lines.sort();
        let tmp = path.with_extension(format!("tmp{}-{}", std::process::id(), TEMPS.fetch_add(1, Ordering::Relaxed)));
        let text = format!("format {FORMAT}\n{}\n", lines.join("\n"));
        if std::fs::write(&tmp, text).is_ok() && std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    /// A measured number (a probe's cost, say) by the same path as
    /// [`Self::choose`]: the memo, the store, else `measure`, stored when `Some`.
    pub(crate) fn value(
        &self,
        key: &TuneKey,
        programs: impl FnOnce() -> u64,
        measure: impl FnOnce() -> Option<u64>,
    ) -> Option<u64> {
        match self.remember(Kind::Value, key, |_| true, || (programs(), ()), |()| measure().map(Entry::Value))? {
            Entry::Value(v) => Some(v),
            _ => None,
        }
    }

    /// The memo, then the stored entry of `kind` that is `valid` under the
    /// fingerprint `programs` makes, else `measure`'s entry of what
    /// `programs` built, stored when there is one and memoized either way.
    /// `None` when nothing measured; that is memoized but never stored.
    pub(crate) fn remember<T>(
        &self,
        kind: Kind,
        key: &TuneKey,
        valid: impl Fn(&Entry) -> bool,
        programs: impl FnOnce() -> (u64, T),
        measure: impl FnOnce(T) -> Option<Entry>,
    ) -> Option<Entry> {
        once(&self.memo, &(kind, key.clone()), || {
            let (programs, built) = programs();
            if self.replays() {
                return self.replayed(kind, key, programs, &valid);
            }
            let line = key.line(kind, programs);
            self.entries(key).remove(&line).filter(&valid).or_else(|| {
                tracing::debug!(target: crate::ops::PLAN, line, "store miss");
                let entry = measure(built)?;
                self.put(key, line, &entry);
                Some(entry)
            })
        })
    }

    /// The indices of `candidates` that ran, fastest first, each with its
    /// device ns: the memo, then the store under the line the programs
    /// `build` makes fingerprint (each built once), else, on a store miss,
    /// those `bench` has not compiled are compiled and every candidate is
    /// [`time`]d through `pass`, their order stored. A candidate whose
    /// programs emit what another's do (an unrolling the schedule folds
    /// away) is timed as its twin: the line still names it, at its twin's ns.
    pub(crate) fn choose<C: crate::ops::config::Tiled + std::fmt::Debug>(
        &self,
        key: &TuneKey,
        candidates: &[C],
        build: impl Fn(C) -> Candidate,
        bench: &mut Bench<C>,
        pass: Pass,
    ) -> Option<Arc<[(usize, u64)]>> {
        let keep = bench.lowered.clone();
        let programs = || {
            let built: Vec<Candidate> = candidates.iter().map(|&c| build(c)).collect();
            let (fingerprint, keys) = self.fingerprint_keeping(&built, Some(&keep));
            (fingerprint, (built, keys))
        };
        let within = |entry: &Entry| matches!(entry, Entry::Choice(timed) if timed.iter().all(|&(i, _, _)| i < candidates.len()));
        let timed = |(built, keys): (Vec<Candidate>, Vec<u64>)| {
            ordered(bench.time(candidates, built, &keys, pass, self)).map(|timed| {
                Entry::Choice(timed.iter().map(|&(i, ns)| (i, candidates[i].code(), ns)).collect::<Vec<_>>().into())
            })
        };
        match self.remember(Kind::Choice, key, within, programs, timed)? {
            Entry::Choice(timed) => Some(timed.iter().map(|&(i, _, ns)| (i, ns)).collect()),
            _ => None,
        }
    }

    /// The indices `admit` finds for the ranked list `key` names (`None`:
    /// nothing compiled to judge) by the same path as [`Self::choose`]: the
    /// memo, the store under the line `programs` fingerprints, else `admit`,
    /// stored when it judged. What compiles without spilling is the
    /// compiler's and the programs' alone, both in the line, so a later
    /// process compiles nothing but its choice.
    pub(crate) fn admitted<C: crate::ops::config::Tiled>(
        &self,
        key: &TuneKey,
        list: &[C],
        programs: impl FnOnce() -> u64,
        admit: impl FnOnce() -> Option<Vec<usize>>,
    ) -> Option<Arc<[usize]>> {
        let judged =
            |()| admit().map(|at| Entry::Admitted(at.iter().map(|&i| (i, list[i].code())).collect::<Vec<_>>().into()));
        match self.remember(Kind::Admitted, key, |_| true, || (programs(), ()), judged)? {
            Entry::Admitted(at) => Some(at.iter().map(|&(i, _)| i).collect()),
            _ => None,
        }
    }

    /// The index `find` names as the first clean config of the ranked list
    /// `key` names (`None`: nothing compiled), by the same path as
    /// [`Self::admitted`] under its own kind, so a list's full admission and
    /// its first clean config are each remembered once.
    pub(crate) fn first<C: crate::ops::config::Tiled>(
        &self,
        key: &TuneKey,
        list: &[C],
        programs: impl FnOnce() -> u64,
        find: impl FnOnce() -> Option<usize>,
    ) -> Option<usize> {
        let found = |()| find().and_then(|at| list.get(at).map(|c| Entry::Admitted([(at, c.code())].into())));
        match self.remember(Kind::First, key, |_| true, || (programs(), ()), found)? {
            Entry::Admitted(at) => at.first().map(|&(i, _)| i),
            _ => None,
        }
    }
}

/// The candidates that ran, fastest first, each with its ns; `None` when
/// none did.
pub(crate) fn ordered(ns: Vec<Option<u64>>) -> Option<Arc<[(usize, u64)]>> {
    let mut ran: Vec<(u64, usize)> = ns.into_iter().enumerate().filter_map(|(i, ns)| Some((ns?, i))).collect();
    ran.sort_unstable();
    (!ran.is_empty()).then(|| ran.into_iter().map(|(ns, i)| (i, ns)).collect())
}

/// A candidate compiled on scratch buffers: its plans (`None` when one
/// failed to build) and its kernels' compiled resources where the backend
/// reports them.
pub(crate) struct Built {
    plans: Option<Vec<Scratch>>,
    resources: Vec<KernelResources>,
}

/// What compiling a candidate showed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Compiled {
    Failed,
    /// A kernel [`spilled`]: admitted only where every candidate does, timed
    /// among a tile's pipelines.
    Spilled,
    Clean,
}

impl Built {
    pub(crate) fn verdict(&self) -> Compiled {
        match &self.plans {
            None => Compiled::Failed,
            Some(_) if spilled(&self.resources) => Compiled::Spilled,
            Some(_) => Compiled::Clean,
        }
    }
}

/// Whether a kernel of `resources` spilled to scratch; a backend reporting
/// nothing has not.
pub(crate) fn spilled(resources: &[KernelResources]) -> bool {
    resources.iter().any(|r| r.scratch_bytes.is_some_and(|b| b > 0))
}

/// Candidates compiled once on one scratch pool and kept: admitted as the
/// shortlist's search asks, then timed by the passes that ask for them, none
/// compiled or lowered twice (a fingerprint's lowerings are kept for the
/// compile). One whose programs emit what another's do compiles and times no
/// more: it is that one's twin, its verdict and time the twin's ([`Emitted`]).
/// Compiles run on the rayon pool ([`streamed`]), each killed at
/// [`COMPILE_TIMEOUT`]; while candidates are timed, the rest compile.
pub(crate) struct Bench<C> {
    pool: ScratchPool,
    pub(crate) built: Vec<(C, Built)>,
    pub(crate) lowered: Arc<Lowerings>,
    pub(crate) emitted: Emitted<C>,
}

/// What the candidates compiled so far emit, by digest: the first config to
/// emit a kernel, and per twin of it skipped the config that compiled for
/// the twin. A digest maps a candidate's programs whole, so candidates of
/// one shape cannot share another's; a config, not a `built` index, names
/// the compiler, a timing's swap-remove reordering them.
pub(crate) struct Emitted<C> {
    pub(crate) first: HashMap<u64, C>,
    /// A skipped twin and the config that compiled for it.
    pub(crate) twin: HashMap<C, C>,
}

impl<C> Default for Emitted<C> {
    fn default() -> Self {
        Self { first: HashMap::new(), twin: HashMap::new() }
    }
}

impl<C> Default for Bench<C> {
    fn default() -> Self {
        Self { pool: ScratchPool::default(), built: vec![], lowered: Arc::default(), emitted: Default::default() }
    }
}

/// Search bound: a candidate's compile past this is killed and the
/// candidate rejected, as one that fails (its lowering and compiler runs
/// together; Tinygrad's BEAM gives one 10 s). On gfx1201 the 16384² d128
/// k64 nearest's queue-32 tiles spill at 256 VGPRs after 62–77 s of clang
/// each, which was most of that shape's admission, while every clean one of
/// nine nearest shapes compiled within 9 s at eight compiles at once (the
/// slowest clang seen beside another process's load: 21 s).
pub(crate) const COMPILE_TIMEOUT: Duration = Duration::from_secs(30);

/// Compiles of a [`Bench`], `width` at a time, keyed by `K`: what compiled,
/// what it emitted, and how long it took.
type Compiling<'a, K> = Stream<'a, (K, Candidate), (K, Built, Option<u64>, Duration)>;

impl<C: Copy + PartialEq + Eq + std::hash::Hash + std::fmt::Debug + Send> Bench<C> {
    fn find(&self, c: C) -> Option<usize> {
        self.built
            .iter()
            .position(|(b, _)| *b == c)
            .or_else(|| self.emitted.twin.get(&c).and_then(|&of| self.built.iter().position(|(b, _)| *b == of)))
    }

    /// `body` over a stream compiling candidates on this bench's pool and
    /// lowerings, `width` at once.
    fn compiling<K: Send, T>(
        &self,
        store: &TuneStore,
        width: usize,
        body: impl FnOnce(&mut Compiling<'_, K>) -> T,
    ) -> T {
        let (pool, lowered) = (&self.pool, &self.lowered);
        let run = |(key, programs): (K, Candidate)| {
            let start = std::time::Instant::now();
            let (built, emitted) = svod_runtime::clang::with_compile_deadline(start + COMPILE_TIMEOUT, || {
                compile_one(programs, pool, Some((store, lowered)))
            });
            (key, built, emitted, start.elapsed())
        };
        streamed(width, run, body)
    }

    /// What compiling each of `batch` (the programs `build` makes) showed,
    /// those not compiled before compiled in parallel. A candidate whose
    /// programs emit what one already built's (or an earlier one of the
    /// batch's) do is its twin: it compiles nothing, its verdict the twin's.
    pub(crate) fn admit(&mut self, batch: &[C], build: impl Fn(C) -> Candidate, store: &TuneStore) -> Vec<Compiled> {
        let missing: Vec<C> = distinct(batch.iter().copied().filter(|&c| self.find(c).is_none()));
        let mut keys: HashMap<u64, C> = self.emitted.first.clone();
        let mut fresh: Vec<(C, Candidate)> = vec![];
        for &c in &missing {
            let programs = build(c);
            match store.emitted_key(&programs) {
                Some(k) if keys.contains_key(&k) => {
                    self.emitted.twin.insert(c, keys[&k]);
                }
                Some(k) => {
                    keys.insert(k, c);
                    fresh.push((c, programs));
                }
                None => fresh.push((c, programs)),
            }
        }
        let done = self.compiling(store, threads(), |stream| {
            fresh.into_iter().for_each(|job| stream.push(job));
            std::iter::from_fn(|| stream.next()).collect::<Vec<_>>()
        });
        let (mut built, mut emitted) = (std::mem::take(&mut self.built), std::mem::take(&mut self.emitted));
        for (cfg, built_one, key, took) in done {
            keep(&mut built, &mut emitted, cfg, built_one, key, took);
        }
        (self.built, self.emitted) = (built, emitted);
        batch.iter().map(|&c| self.built[self.find(c).expect("compiled")].1.verdict()).collect()
    }

    /// The tuner's shortlist of `ranked` ([`crate::ops::config::shortlist`]),
    /// each config compiled as soon as the search wants it. A config whose
    /// programs emit what one already built's does is its twin (its verdict
    /// waits on the compile of the same kernel, in flight or done); one held
    /// for a kernel still compiling answers as it lands.
    pub(crate) fn shortlist(
        &mut self,
        ranked: &[C],
        build: impl Fn(C) -> Candidate,
        store: &TuneStore,
    ) -> Option<Vec<usize>>
    where
        C: crate::ops::config::Tiled,
    {
        struct Search<'s, 'b, C, B> {
            stream: &'s mut Compiling<'b, (usize, C)>,
            bench: &'s mut Vec<(C, Built)>,
            emitted: &'s mut Emitted<C>,
            build: B,
            store: &'s TuneStore,
            /// Compiles started per emitted kernel, not landed yet.
            started: HashMap<u64, C>,
            /// Starts held for a kernel compiling: their configs answer as
            /// its compile lands.
            held: Vec<(usize, u64, C)>,
            /// Verdicts a twin's compile made, ready to hand back.
            ready: std::collections::VecDeque<(usize, Compiled)>,
        }
        impl<C: Copy + PartialEq + Eq + std::hash::Hash + std::fmt::Debug, B: Fn(C) -> Candidate>
            crate::ops::config::Compiles<C> for Search<'_, '_, C, B>
        {
            fn start(&mut self, at: usize, config: C) {
                let programs = (self.build)(config);
                let Some(k) = self.store.emitted_key(&programs) else {
                    self.stream.push(((at, config), programs));
                    return;
                };
                if let Some(&of) = self.emitted.first.get(&k) {
                    let Some((_, built)) = self.bench.iter().find(|(b, _)| *b == of) else {
                        self.stream.push(((at, config), programs));
                        return;
                    };
                    self.emitted.twin.insert(config, of);
                    self.ready.push_back((at, built.verdict()));
                    return;
                }
                match self.started.insert(k, config) {
                    // The same kernel compiling: its verdict is this one's.
                    Some(of) => {
                        let _ = self.started.insert(k, of);
                        self.held.push((at, k, config));
                    }
                    None => self.stream.push(((at, config), programs)),
                }
            }

            fn idle(&self) -> bool {
                self.stream.idle()
            }

            fn finished(&mut self) -> Option<(usize, Compiled)> {
                if let Some((at, verdict)) = self.ready.pop_front() {
                    return Some((at, verdict));
                }
                let ((at, cfg), built, key, took) = self.stream.next()?;
                let verdict = keep(self.bench, self.emitted, cfg, built, key, took);
                if let Some(k) = key {
                    self.started.remove(&k);
                    let (answering, held): (Vec<_>, Vec<_>) =
                        std::mem::take(&mut self.held).into_iter().partition(|&(_, held, _)| held == k);
                    for (at, _, twin) in answering {
                        self.emitted.twin.insert(twin, cfg);
                        self.ready.push_back((at, verdict));
                    }
                    self.held = held;
                }
                Some((at, verdict))
            }
        }
        let (mut built, mut emitted) = (std::mem::take(&mut self.built), std::mem::take(&mut self.emitted));
        let found = self.compiling(store, threads(), |stream| {
            crate::ops::config::shortlist(
                ranked,
                &mut Search {
                    stream,
                    bench: &mut built,
                    emitted: &mut emitted,
                    build,
                    store,
                    started: HashMap::new(),
                    held: vec![],
                    ready: Default::default(),
                },
            )
        });
        (self.built, self.emitted) = (built, emitted);
        found
    }

    /// Each of `candidates`' best device time, timed in their order (the
    /// ranking's, so a good time is soon the bar the [`HOPELESS`] ones miss):
    /// one a [`emitted`](Emitted) key shares with another times once, as the
    /// first of them, its twins at its ns. Those not compiled yet compile
    /// from `programs`, in order, on all but one of the pool's threads while
    /// the ones ready are timed.
    pub(crate) fn time(
        &mut self,
        candidates: &[C],
        mut programs: Vec<Candidate>,
        keys: &[u64],
        pass: Pass,
        store: &TuneStore,
    ) -> Vec<Option<u64>> {
        let mut firsts: HashMap<&u64, usize> = HashMap::new();
        let of: Vec<usize> = keys.iter().enumerate().map(|(i, k)| *firsts.entry(k).or_insert(i)).collect();
        let runs: Vec<usize> = (0..candidates.len()).filter(|&i| of[i] == i).collect();
        let at: Vec<C> = runs.iter().map(|&i| candidates[i]).collect();
        let programs: Vec<Candidate> = runs.iter().map(|&i| std::mem::take(&mut programs[i])).collect();
        let ns = self.timed(&at, programs, pass, store);
        let mut out = vec![None; candidates.len()];
        for (i, ns) in runs.into_iter().zip(ns) {
            out[i] = ns;
        }
        for (i, &of) in of.iter().enumerate() {
            out[i] = out[of];
        }
        out
    }

    /// [`Self::time`] of candidates known apart by what they emit.
    fn timed(&mut self, candidates: &[C], programs: Vec<Candidate>, pass: Pass, store: &TuneStore) -> Vec<Option<u64>> {
        let mut slots: Vec<Option<Built>> = candidates.iter().map(|&c| self.take(c)).collect();
        let missing: Vec<(usize, Candidate)> =
            programs.into_iter().enumerate().filter(|(i, _)| slots[*i].is_none()).collect();
        // Their scratch buffers filled now: a fill is a kernel, which would
        // run beside the ones timed.
        for (_, candidate) in &missing {
            for (prog, _) in candidate {
                self.pool.buffers(&prog.params);
            }
        }
        let ns = self.compiling(store, threads().saturating_sub(1), |stream| {
            missing.into_iter().for_each(|job| stream.push(job));
            let arrive = || stream.next().map(|(i, built, _, _)| (i, built));
            time_with(&mut slots, arrive, Some(HOPELESS), pass, true)
        });
        for (&c, built) in candidates.iter().zip(slots) {
            self.built.extend(built.map(|b| (c, b)));
        }
        ns
    }

    /// The candidate's compile, taken out of `built` for a timing: a twin's
    /// held under the config that compiled for it, which the timing's
    /// candidate then holds (its Built lands under its own config, and what
    /// compiled the kernel names it).
    fn take(&mut self, c: C) -> Option<Built> {
        let at = self.find(c)?;
        let (held, built) = self.built.swap_remove(at);
        if held != c {
            for of in self.emitted.twin.values_mut() {
                if *of == held {
                    *of = c;
                }
            }
            for of in self.emitted.first.values_mut() {
                if *of == held {
                    *of = c;
                }
            }
        }
        Some(built)
    }
}

/// Keeps what compiling `cfg` built in `bench` and logs it; its verdict.
/// What it emitted, where the compile learned it, records `cfg` as the first
/// to emit that kernel.
fn keep<C: std::fmt::Debug + Clone>(
    bench: &mut Vec<(C, Built)>,
    emitted: &mut Emitted<C>,
    cfg: C,
    built: Built,
    key: Option<u64>,
    took: Duration,
) -> Compiled {
    let verdict = built.verdict();
    let registers: Vec<_> = built.resources.iter().map(|r| (r.vgprs, r.agprs)).collect();
    let scratch: Vec<_> = built.resources.iter().map(|r| r.scratch_bytes).collect();
    tracing::debug!(target: crate::ops::PLAN, ?cfg, ?verdict, ?registers, ?scratch, ?took, "compiled");
    if verdict == Compiled::Failed && took >= COMPILE_TIMEOUT {
        tracing::info!(target: crate::ops::PLAN, ?cfg, ?took, "compile timed out: rejected");
    }
    if let Some(k) = key {
        emitted.first.entry(k).or_insert_with(|| cfg.clone());
    }
    bench.push((cfg, built));
    verdict
}

/// The rayon pool's threads, the most compiles a stream runs at once.
fn threads() -> usize {
    rayon::current_num_threads()
}

fn distinct<C: PartialEq>(items: impl Iterator<Item = C>) -> Vec<C> {
    let mut out: Vec<C> = vec![];
    for c in items {
        if !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

/// Builds each candidate's plans on `pool`'s scratch buffers at capacity
/// (every runtime variable bound to its maximum) and reads its kernels'
/// [`resources`]. The programs compile in parallel, on workers carrying the
/// caller's default device and origin scope.
pub(crate) fn compile(candidates: Vec<Candidate>, pool: &ScratchPool) -> Vec<Built> {
    in_parallel(candidates, |programs| compile_one(programs, pool, None).0)
}

/// One candidate's [`compile`], its programs lowered through `lowered`
/// (`store`'s fingerprint memo and the lowerings a fingerprint kept) where
/// given: a program that does not lower fails the candidate there, lowered
/// no further. What its programs emit, where a store lowered them, keys the
/// candidate for its twins; `None` where one did not lower (such a candidate
/// fails, and nothing emits).
fn compile_one(
    programs: Candidate,
    pool: &ScratchPool,
    lowered: Option<(&TuneStore, &Lowerings)>,
) -> (Built, Option<u64>) {
    let mut emitted: Vec<u64> = vec![];
    let plans: Option<Vec<Scratch>> = programs
        .into_iter()
        .map(|(p, l)| match lowered {
            Some((store, keep)) => {
                let (digest, body) = store.lowered(&p, &l, keep)?;
                emitted.push(digest);
                scratch_plan_of(p, &l, pool, Some(body))
            }
            None => scratch_plan(p, &l, pool),
        })
        .collect();
    let key = plans.as_ref().filter(|_| !emitted.is_empty()).map(|_| digest(&emitted));
    let resources = plans.iter().flatten().flat_map(|(plan, _)| resources(plan)).collect();
    (Built { plans, resources }, key)
}

/// Held across every [`time`] in the process, the probes' and the tuner's
/// alike: two measurements sharing the device would time each other.
static DEVICE: Mutex<()> = Mutex::new(());

/// Each built candidate's best device time in ns, its programs' summed;
/// `None` where one failed to build or run. The device runs them one at a
/// time, after the clock is lifted, under the process's one timing lock.
pub(crate) fn time(built: &mut [Built]) -> Vec<Option<u64>> {
    let mut slots: Vec<Option<&mut Built>> = built.iter_mut().map(Some).collect();
    time_with(&mut slots, || None, None, Pass::Whole, false)
}

/// [`time`] through [`rounds`] under `hopeless` and `pass` of the
/// candidates in `slots`, an empty one filled from `arrive` (the compiles
/// that land, each with its candidate's index) as timing reaches it: a
/// candidate whose first timed run passes that factor of the best so far
/// keeps that run's time.
fn time_with<B: BorrowMut<Built>>(
    slots: &mut [Option<B>],
    mut arrive: impl FnMut() -> Option<(usize, B)>,
    hopeless: Option<f64>,
    pass: Pass,
    budgeted: bool,
) -> Vec<Option<u64>> {
    let _device = DEVICE.lock().unwrap_or_else(PoisonError::into_inner);
    let count = slots.len();
    // Whether a candidate's plans ran once (`None`: not tried yet).
    let mut ran: Vec<Option<bool>> = vec![None; count];
    let Some(first) = (0..count).find(|&i| ready(i, slots, &mut arrive, &mut ran).is_some()) else {
        return vec![None; count];
    };
    let plan = ready(first, slots, &mut arrive, &mut ran).expect("ran")[0];
    warm_clock(CLOCK_WARMUP, || run(plan));
    // A plan's last timed run sizes its sustain, one timed run first.
    let mut last: Vec<Vec<Option<Duration>>> = vec![vec![]; count];
    let mut time = |i: usize, abort: Option<Duration>, leader: bool| {
        let plans = ready(i, slots, &mut arrive, &mut ran)?;
        let last = &mut last[i];
        last.resize(plans.len(), None);
        // The first timed run, where this call takes it.
        let mut fresh = vec![None; plans.len()];
        for ((plan, last), fresh) in plans.iter().zip(last.iter_mut()).zip(&mut fresh).filter(|((_, l), _)| l.is_none())
        {
            *last = Some(run(plan)?);
            *fresh = *last;
        }
        let first: Duration = last.iter().flatten().sum();
        if abort.is_some_and(|abort| first > abort) {
            return Some(first);
        }
        let (runs, sustained) = match budgeted && !leader {
            true => (runs(first), first < SUSTAIN),
            false => (RUNS, true),
        };
        if sustained {
            sustain(&plans, SUSTAIN, first);
        }
        plans
            .iter()
            .zip(last.iter_mut())
            .zip(fresh)
            .map(|((plan, last), fresh)| {
                // Unsustained, the first run is as good as the next ones.
                let mut timed = vec![fresh.or_else(|| run(plan))?];
                if sustained {
                    // Run until the minimum is consistent: the budget's
                    // runs the floor, three stable ones settle it.
                    let (mut stable, mut least) = (0, timed[0]);
                    while timed.len() < MOST_RUNS && (timed.len() < runs || stable < STABLE_RUNS) {
                        let r = run(plan)?;
                        stable = if r.as_secs_f64() < least.as_secs_f64() * STEADY { 0 } else { stable + 1 };
                        least = least.min(r);
                        timed.push(r);
                    }
                }
                let t = best(timed.into_iter().map(Some))?;
                *last = Some(t);
                Some(t)
            })
            .sum::<Option<Duration>>()
    };
    rounds(count, hopeless, pass, &mut time).into_iter().map(|t| t.map(|t| t.as_nanos() as u64)).collect()
}

/// Candidate `i`'s plans once they ran a first time (untimed), its
/// compile awaited through `arrive` where its slot is empty; `None` where
/// it failed to build or run.
fn ready<'s, B: BorrowMut<Built>>(
    i: usize,
    slots: &'s mut [Option<B>],
    arrive: &mut impl FnMut() -> Option<(usize, B)>,
    ran: &mut [Option<bool>],
) -> Option<Vec<&'s ExecutionPlan>> {
    while slots[i].is_none() {
        let (at, built) = arrive()?;
        slots[at] = Some(built);
    }
    let plans = slots[i].as_mut()?.borrow_mut().plans.as_mut()?;
    let ok = *ran[i].get_or_insert_with(|| {
        plans.iter_mut().all(|(plan, vars)| {
            let vars: Vec<(&str, i64)> = vars.iter().map(|(n, v)| (n.as_str(), *v)).collect();
            plan.execute_with_vars(&vars).is_ok()
        })
    });
    ok.then(|| plans.iter().map(|(plan, _)| plan).collect())
}

/// Each candidate's minimum over the timed rounds: one round of every
/// candidate, then [`ROUNDS`]` - 1` more of the [`CONTENDER`]s of the first
/// round's best (under [`CUT_FROM`], those within `hopeless` of it, else
/// every candidate), each round past the second left to those within
/// [`SETTLED`] of the best (from [`CUT_FROM`]), then, where more than
/// two contenders ran them, [`ROUNDS`] more of the two leaders, whose gap
/// is often within a round's drift (two contenders have run every round
/// together already); a [`Pass::Screen`] stops after the second round.
/// The first round runs in the given order, the later ones in turn
/// backwards and forwards: a candidate runs slower after some others (on
/// sm_86 a nearest tile ran 4 % slower after its tile's other pipelines than
/// after a larger tile, timed in one order or the other), and its minimum
/// over both neighbours' runs keeps that out of the choice.
/// `time` times a candidate; in the first round its
/// second argument is the bar, `hopeless` times the best time so far, past
/// which the candidate's first run may stand as its time; its third says a
/// leaders' round (timed at full [`RUNS`]). `None` where a candidate failed
/// to run.
pub(crate) fn rounds(
    count: usize,
    hopeless: Option<f64>,
    pass: Pass,
    mut time: impl FnMut(usize, Option<Duration>, bool) -> Option<Duration>,
) -> Vec<Option<Duration>> {
    let mut lead: Option<Duration> = None;
    let mut best = round_robin_min(count, 1, None, |i| {
        let t = time(i, hopeless.zip(lead).map(|(k, lead)| lead.mul_f64(k)), false);
        lead = t.map(|t| lead.map_or(t, |lead| lead.min(t))).or(lead);
        t
    });
    let Some(lead) = best.iter().flatten().min().copied() else { return best };
    let bound = if count >= CUT_FROM { Some(CONTENDER) } else { hopeless }.map(|k| lead.mul_f64(k));
    let contenders: Vec<usize> =
        (0..count).filter(|&i| best[i].is_some_and(|t| bound.is_none_or(|bound| t <= bound))).collect();
    tracing::debug!(first = ?best, ?contenders, "tuner's first round");
    let (mut live, mut later) = (contenders.clone(), vec![None; count]);
    let screened = pass == Pass::Screen;
    for round in 1..if screened { 2 } else { ROUNDS } {
        let order: Vec<usize> = if round % 2 == 1 { live.iter().rev().copied().collect() } else { live.clone() };
        for &i in &order {
            let t = time(i, None, false);
            later[i] = t.map(|t| later[i].map_or(t, |l: Duration| l.min(t)));
            best[i] = t.and_then(|t| best[i].map(|b| b.min(t)));
        }
        let settled = best.iter().flatten().min().map(|lead| lead.mul_f64(SETTLED)).filter(|_| count >= CUT_FROM);
        let near = |i: usize| best[i].is_some_and(|t| settled.is_none_or(|settled| t <= settled));
        live.retain(|&i| later[i].is_some_and(|t| bound.is_none_or(|bound| t <= bound)) && near(i));
    }
    if contenders.len() > 2 && !screened {
        let mut leaders: Vec<usize> = (0..count).filter(|&i| best[i].is_some()).collect();
        leaders.sort_by_key(|&i| best[i]);
        leaders.truncate(2);
        if let [first, second] = leaders[..] {
            let more = round_robin_min(2, ROUNDS, None, |j| time(leaders[j], None, true));
            for (&i, more) in leaders.iter().zip(more) {
                best[i] = more.and_then(|m| best[i].map(|b| b.min(m)));
            }
            tracing::debug!(leader = first, runner_up = second, times = ?(best[first], best[second]), "tuner's leaders");
        }
    }
    tracing::debug!(?best, "tuner's rounds");
    best
}

/// A compiled plan over scratch buffers and the variable bindings it runs at.
type Scratch = (ExecutionPlan, Vec<(String, i64)>);

/// The scratch buffers every candidate of a measurement runs on: one per
/// `(dtype, capacity, ordinal within a program)`, so a program's equal-sized
/// parameters stay distinct while the candidates share. A cache-mode attention
/// would otherwise hold a whole key and value cache per candidate at once.
#[derive(Default)]
pub(crate) struct ScratchPool(Mutex<HashMap<(ScalarDType, usize, usize), Tensor>>);

impl ScratchPool {
    pub(crate) fn buffers(&self, params: &[Param]) -> Vec<Tensor> {
        let mut seen: HashMap<(ScalarDType, usize), usize> = HashMap::new();
        let mut pool = self.0.lock().expect("scratch pool");
        params
            .iter()
            .map(|p| {
                let ordinal = seen.entry((p.dtype, p.elems)).or_insert(0);
                let key = (p.dtype, p.elems, std::mem::replace(ordinal, *ordinal + 1));
                pool.entry(key)
                    .or_insert_with(|| {
                        // Written here: an unrealized buffer is allocated by
                        // every plan that reads it, one per candidate. The
                        // fill is prepared as a candidate is, in passing: it
                        // moves no kernel-name counter, so a model's kernel
                        // of its shape, prepared after a tune, keeps the name
                        // and the object it has in a process that tuned
                        // nothing.
                        let t = scratch(p);
                        let mut config = PrepareConfig::from_env();
                        config.ephemeral = true;
                        t.realize_with(&config).expect("a scratch buffer");
                        t
                    })
                    .clone()
            })
            .collect()
    }
}

/// A scratch buffer's unrealized fill: a float parameter holds values in
/// `[-1, 1)`, drawn on the device, so a card that clocks down under real
/// data (the RX 9070 XT runs the 4096³ GEMM 15 % slower on random operands
/// than on zeros, which toggle no bit) is timed as it will run; an integer
/// parameter holds zeros, since a probe reads it as the slot to read next.
fn scratch(p: &Param) -> Tensor {
    if p.dtype.is_float() {
        let unit = Tensor::rand_with(&[p.elems], DType::Float32, default_device()).expect("a scratch fill");
        let spread = unit.try_mul(2.0f32).and_then(|t| t.try_sub(1.0f32)).expect("a scratch fill");
        spread.cast(p.dtype.into()).contiguous()
    } else {
        Tensor::zeros(&[p.elems], p.dtype.into()).contiguous()
    }
}

/// A candidate's plan on `pool`'s buffers: lowered and prepared for this
/// measurement alone, kept in no memo or cache (it runs under scratch
/// placeholders no op's call keys, and a shortlist of them per shape would
/// otherwise stay for the process).
pub(crate) fn scratch_plan(prog: Program, lowering: &Lowering, pool: &ScratchPool) -> Option<Scratch> {
    scratch_plan_of(prog, lowering, pool, None)
}

/// [`scratch_plan`] of `body`, the program lowered already
/// ([`crate::lower::standalone`]) where given.
fn scratch_plan_of(prog: Program, lowering: &Lowering, pool: &ScratchPool, body: Option<Arc<UOp>>) -> Option<Scratch> {
    let vars: Vec<(String, i64)> = prog.vars.iter().map(|v| (v.name.clone(), v.max)).collect();
    let name = prog.name.clone();
    let buffers = pool.buffers(&prog.params);
    let tensors: Vec<&Tensor> = buffers.iter().collect();
    let out = match body {
        Some(body) => launch::graph_launch_lowered(&prog, lowering, body, &tensors)
            .map(|mut outs| outs.swap_remove(launch::first_output(&prog))),
        None => launch::graph_launch_once(prog, lowering, &tensors),
    }
    .ok()?;
    let mut config = PrepareConfig::from_env();
    config.ephemeral = true;
    // A program past what the scheduler's rewrites hold (a queued list two
    // lanes wide 32 deep built 518k nodes on gfx1201, past the rewrite
    // engine's stack) panics there: the candidate fails to build, as one
    // the compiler refuses, rather than taking the tuning down.
    let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| out.prepare_with(&config)));
    let plan = prepared.map_err(|_| tracing::warn!(kernel = name, "a candidate's graph failed to schedule")).ok()?;
    Some((plan.ok()?, vars))
}

/// Runs a candidate's `plans` back to back, in order, unsynchronized, for
/// about `budget` of device time at `per_run` a run of them all. Launches
/// queue without waiting, so a spin of host time would queue a thousand
/// runs of a short kernel and every timed run after it would wait for the
/// queue to drain. The candidate is sustained whole, not each plan for the
/// budget: a split's merge alone queued a thousand microsecond launches,
/// and the candidates timed after it ran slow (on gfx1201 an unsplit
/// nearest at depth 768 timed 106–110 TFLOP/s among split ones, 124–126
/// alone or with each candidate sustained whole, so the tuner never chose
/// it).
fn sustain(plans: &[&ExecutionPlan], budget: Duration, per_run: Duration) {
    let launches = budget.as_nanos().div_ceil(per_run.as_nanos().max(1)).clamp(1, SUSTAIN_LAUNCHES);
    for _ in 0..launches {
        if plans.iter().any(|plan| plan.execute().is_err()) {
            break;
        }
    }
}

/// `f` over `items` on the rayon pool, each worker running under the
/// caller's default device, origin scope and [`scoped`] tuning (all per
/// thread), so what it builds is what the caller would have built.
pub(crate) fn in_parallel<T: Send, R: Send>(items: Vec<T>, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    use rayon::prelude::*;
    let context = Context::here();
    items.into_par_iter().map(|item| context.enter(|| f(item))).collect()
}

/// What a worker takes over from the thread that hands it work: the
/// default device, origin scope and [`scoped`] tuning (all per thread).
struct Context {
    device: DeviceSpec,
    origin: Option<svod_ir::origin::OriginId>,
    capture: bool,
    tuning: Option<bool>,
}

impl Context {
    fn here() -> Self {
        let (origin, capture) = (svod_ir::origin::current(), svod_ir::origin::enabled());
        Self { device: default_device(), origin, capture, tuning: SCOPED.get() }
    }

    fn enter<R>(&self, f: impl FnOnce() -> R) -> R {
        with_default_device(self.device.clone(), || {
            let _capture = svod_ir::origin::capture_for_thread(self.capture);
            let _scope = svod_ir::origin::install(self.origin);
            let _tuning = Restore(SCOPED.replace(self.tuning));
            f()
        })
    }
}

/// `body` handed a [`Stream`] whose jobs `run` takes on the rayon pool, at
/// most `width` at once, in the order queued, each under this thread's
/// [`Context`]; a job's panic resumes on this thread. Jobs still running
/// when `body` returns are waited for, their results dropped.
pub(crate) fn streamed<J: Send, R: Send, T>(
    width: usize,
    run: impl Fn(J) -> R + Sync,
    body: impl FnOnce(&mut Stream<'_, J, R>) -> T,
) -> T {
    let context = Context::here();
    let (run, context) = (&run, &context);
    let (sent, results) = std::sync::mpsc::channel();
    rayon::in_place_scope(|scope| {
        let spawn = move |job: J| {
            let sent = sent.clone();
            scope.spawn(move |_| {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| context.enter(|| run(job))));
                let _ = sent.send(result);
            });
        };
        let mut stream =
            Stream { spawn: Box::new(spawn), results, queue: Default::default(), flight: 0, width: width.max(1) };
        body(&mut stream)
    })
}

/// The jobs of a [`streamed`] body.
pub(crate) struct Stream<'a, J, R> {
    spawn: Box<dyn FnMut(J) + 'a>,
    results: std::sync::mpsc::Receiver<std::thread::Result<R>>,
    queue: std::collections::VecDeque<J>,
    flight: usize,
    width: usize,
}

impl<J, R> Stream<'_, J, R> {
    /// Whether a job pushed now would start at once.
    pub(crate) fn idle(&self) -> bool {
        self.queue.is_empty() && self.flight < self.width
    }

    pub(crate) fn push(&mut self, job: J) {
        self.queue.push_back(job);
        self.pump();
    }

    fn pump(&mut self) {
        while self.flight < self.width
            && let Some(job) = self.queue.pop_front()
        {
            (self.spawn)(job);
            self.flight += 1;
        }
    }

    /// A job's result as it lands, waiting for one; `None` when no job is
    /// queued or running. A pool thread waiting here runs the pool's jobs.
    pub(crate) fn next(&mut self) -> Option<R> {
        if self.flight == 0 {
            return None;
        }
        let result = loop {
            if let Ok(result) = self.results.try_recv() {
                break result;
            }
            match rayon::yield_now() {
                Some(rayon::Yield::Executed) => {}
                Some(rayon::Yield::Idle) => {
                    if let Ok(result) = self.results.recv_timeout(Duration::from_millis(1)) {
                        break result;
                    }
                }
                None => break self.results.recv().expect("a job's sender lives until it sends"),
            }
        };
        self.flight -= 1;
        self.pump();
        Some(result.unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
    }
}

/// The fastest of a candidate's timed runs, dropping any under half their
/// median (a lost stamp); `None` only when no run was timed.
pub(crate) fn best(runs: impl Iterator<Item = Option<Duration>>) -> Option<Duration> {
    let mut runs: Vec<Duration> = runs.flatten().collect();
    runs.sort_unstable();
    let median = *runs.get(runs.len() / 2)?;
    runs.into_iter().find(|&run| 2 * run >= median)
}

/// One run's device time, its kernels' summed: a candidate is a kernel plus
/// its merge, which the longest alone would hide. `None` when the run fails
/// or the device did not stamp every kernel (see [`stamped`]).
pub(crate) fn run(plan: &ExecutionPlan) -> Option<Duration> {
    stamped(plan.execute_profiled().ok()?.iter().map(|k| (k.gpu_start_ns, k.gpu_end_ns)))
}

/// The sum of the `(start, end)` stamps of a run's kernels; `None` when there
/// are none or any lost one, since a run without it would read as faster.
pub(crate) fn stamped(stamps: impl IntoIterator<Item = (Option<u64>, Option<u64>)>) -> Option<Duration> {
    let ns: Vec<u64> = stamps.into_iter().map(|(start, end)| end?.checked_sub(start?)).collect::<Option<_>>()?;
    (!ns.is_empty()).then(|| Duration::from_nanos(ns.iter().sum()))
}

/// The compiled resources (registers, LDS, scratch) of the plan's kernels
/// whose backend reports them, in schedule order.
pub(crate) fn resources(plan: &ExecutionPlan) -> Vec<KernelResources> {
    plan.kernels().filter_map(|k| k.program.resource_usage()).collect()
}

thread_local! {
    /// The innermost [`scoped`] setting on this thread.
    static SCOPED: Cell<Option<bool>> = const { Cell::new(None) };
}

/// The innermost [`scoped`] setting on this thread, else `SVOD_TK_TUNE`
/// is not `0`.
pub(crate) fn enabled() -> bool {
    SCOPED.get().unwrap_or_else(|| std::env::var("SVOD_TK_TUNE").map_or(true, |v| v != "0"))
}

/// `f` with measuring on or off on this thread, over the environment; the
/// outer setting comes back when `f` returns or unwinds.
pub fn scoped<R>(on: bool, f: impl FnOnce() -> R) -> R {
    let _restore = Restore(SCOPED.replace(Some(on)));
    f()
}

/// Puts a captured [`SCOPED`] setting back on drop.
struct Restore(Option<bool>);

impl Drop for Restore {
    fn drop(&mut self) {
        SCOPED.set(self.0);
    }
}
