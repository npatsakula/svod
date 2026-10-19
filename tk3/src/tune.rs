//! Measured config choice per shape, made at graph build: an op's ranked
//! configs are admitted ([`Bench`]) and, on a store miss, timed; the winner
//! is kept in the process memo and on disk.
//!
//! The store is one line per [`Entry`] (`kind|key numbers`) in
//! `$SVOD_TK3_TUNE_DIR` (else `$XDG_CACHE_HOME/svod/tk3_tune`, else
//! `$HOME/.cache/svod/tk3_tune`), one file per device and crate version. A
//! line's key carries a digest of the device's compiler ([`compiler`]) and a
//! fingerprint of the candidate programs, so a toolchain or kernel change
//! re-measures. An unreadable or unwritable store is a miss, never an error;
//! a line the store cannot read is dropped at its next write.
//! `SVOD_TK3_TUNE=0` or [`set_enabled`] turns candidate timing off: the first
//! admitted candidate is used, while a device's probes still measure.

use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use svod_dtype::default_device::{default_device, with_default_device};
use svod_dtype::{DeviceSpec, ScalarDType};
use svod_runtime::benchmark::{CLOCK_WARMUP, round_robin_min, warm_clock};
use svod_runtime::{ExecutionPlan, KernelResources};
use svod_tensor::{PrepareConfig, Tensor};

use crate::atoms::Target;
use crate::ir::{Param, Program};
use crate::launch;
use crate::lower::Lowering;

/// Timed rounds over every candidate after the warm-up.
const ROUNDS: usize = 4;
/// Rounds past the first go to the candidates within this factor of the
/// first round's best; the rest cannot win (a round's best of [`RUNS`] runs
/// after a sustain sits within a few percent of the candidate's settled
/// time), and a shortlist is mostly losers.
const CONTENDER: f64 = 1.2;
/// Profiled runs per candidate per round.
const RUNS: usize = 5;
/// Device time of back-to-back runs ahead of each candidate's profiled runs,
/// keeping the lifted clock up where their host overhead would let a short
/// kernel's clock sag.
const SUSTAIN: Duration = Duration::from_millis(5);

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
    pub fn new(
        op: &'static str,
        target: &Target,
        compiler: u64,
        dtype: ScalarDType,
        shape: &[usize],
        candidates: &impl Hash,
    ) -> Self {
        let device = format!("{}-{}sm", target.arch.target_name(), target.sms.unwrap_or(0));
        Self { op, device, dtype, shape: shape.to_vec(), candidates: digest(candidates), compiler }
    }

    /// The store line of a `kind` of entry for the candidate programs
    /// fingerprinted as `programs`.
    pub fn line(&self, kind: Kind, programs: u64) -> String {
        let shape: Vec<String> = self.shape.iter().map(usize::to_string).collect();
        let (op, device, dtype, compiler) = (self.op, &self.device, self.dtype, self.compiler);
        let (kind, candidates) = (kind.name(), self.candidates);
        format!("{kind}|{op}|{device}|{dtype:?}|{}|{candidates:016x}|{compiler:016x}|{programs:016x}", shape.join("x"))
    }
}

/// A digest of `spec`'s compiler identity (backend, toolchain, flags) for
/// [`TuneKey::compiler`], once per device; 0 where no device opens.
pub fn compiler(spec: &DeviceSpec) -> u64 {
    static DIGESTS: LazyLock<Once<DeviceSpec, u64>> = LazyLock::new(Mutex::default);
    once(&DIGESTS, spec, || {
        let device = svod_runtime::DEVICE_FACTORIES.device(spec, svod_device::registry::registry());
        device.map_or(0, |device| digest(&device.compiler.cache_key()))
    })
}

/// A candidate's programs, launched in order (a kernel and the merge of
/// its partial results, say).
pub type Candidate = Vec<(Program, Lowering)>;

/// The fingerprint of candidates' programs: their tile programs and lowerings.
pub fn fingerprint(candidates: &[Candidate]) -> u64 {
    digest(&candidates.iter().map(|c| digest(&format!("{c:?}"))).collect::<Vec<_>>())
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
    /// The fastest candidate's index and its device ns.
    Choice { index: usize, ns: u64 },
    /// A measured number: a probe's cost.
    Value(u64),
    /// The ranked indices admission kept.
    Admitted(Arc<[usize]>),
}

/// An [`Entry`]'s kind, leading its line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Choice,
    Value,
    Admitted,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Choice => "choice",
            Kind::Value => "value",
            Kind::Admitted => "admitted",
        }
    }
}

impl Entry {
    /// The entry's numbers as its line holds them after the key.
    fn render(&self) -> String {
        match self {
            Entry::Choice { index, ns } => format!("{index} {ns}"),
            Entry::Value(v) => v.to_string(),
            Entry::Admitted(at) => at.iter().map(usize::to_string).collect::<Vec<_>>().join(" "),
        }
    }

    /// The entry a line of `kind` with `numbers` after its key records.
    fn parse(kind: &str, numbers: &[u64]) -> Option<Self> {
        match (kind, numbers) {
            ("choice", &[index, ns]) => Some(Entry::Choice { index: index as usize, ns }),
            ("value", &[v]) => Some(Entry::Value(v)),
            ("admitted", at) if !at.is_empty() => Some(Entry::Admitted(at.iter().map(|&i| i as usize).collect())),
            _ => None,
        }
    }
}

/// A memo of this process's entries, backed by one file per device under `root`.
#[derive(Debug)]
pub struct TuneStore {
    root: Option<PathBuf>,
    /// The entry per kind and key, `None` where nothing could be measured.
    memo: Once<(Kind, TuneKey), Option<Entry>>,
}

impl TuneStore {
    /// The store rooted at `root` (`None`: memory only).
    pub fn at(root: Option<PathBuf>) -> Self {
        Self { root, memo: Mutex::default() }
    }

    /// The process-wide store per the environment (see the module docs).
    pub fn global() -> &'static Self {
        static STORE: OnceLock<TuneStore> = OnceLock::new();
        STORE.get_or_init(|| {
            let var = |name| std::env::var_os(name).map(PathBuf::from);
            let root = var("SVOD_TK3_TUNE_DIR")
                .or_else(|| var("XDG_CACHE_HOME").map(|c| c.join("svod/tk3_tune")))
                .or_else(|| var("HOME").map(|h| h.join(".cache/svod/tk3_tune")));
            Self::at(root.filter(|dir| std::fs::create_dir_all(dir).is_ok()))
        })
    }

    fn path(&self, key: &TuneKey) -> Option<PathBuf> {
        let name: String = key.device.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        self.root.as_ref().map(|root| root.join(format!("{name}-v{}.txt", env!("CARGO_PKG_VERSION"))))
    }

    /// Every readable `line -> entry` of the key's device file; empty when unreadable.
    pub fn entries(&self, key: &TuneKey) -> HashMap<String, Entry> {
        let Some(text) = self.path(key).and_then(|p| std::fs::read_to_string(p).ok()) else { return HashMap::new() };
        text.lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let line = fields.next()?;
                let numbers: Vec<u64> = fields.map(str::parse).collect::<Result<_, _>>().ok()?;
                Some((line.to_string(), Entry::parse(line.split('|').next()?, &numbers)?))
            })
            .collect()
    }

    /// Re-read, merge and replace the file atomically, so concurrent writers
    /// lose at most each other's newest line, never the file.
    fn put(&self, key: &TuneKey, line: String, entry: &Entry) {
        let Some(path) = self.path(key) else { return };
        let mut entries = self.entries(key);
        entries.insert(line, entry.clone());
        let mut lines: Vec<String> = entries.iter().map(|(line, entry)| format!("{line} {}", entry.render())).collect();
        lines.sort();
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        if std::fs::write(&tmp, lines.join("\n") + "\n").is_ok() && std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    /// The winner among `count` candidates: the memo, then the store under
    /// the line `programs` fingerprints, else `measure` times every candidate
    /// (device ns, `None` where one cannot run) and the fastest is stored.
    /// `None` when nothing measured; that is memoized but never stored.
    pub fn select(
        &self,
        key: &TuneKey,
        count: usize,
        programs: impl FnOnce() -> u64,
        measure: impl FnOnce() -> Vec<Option<u64>>,
    ) -> Option<usize> {
        let within = |entry: &Entry| matches!(entry, Entry::Choice { index, .. } if *index < count);
        let timed = |()| fastest(measure()).map(|(index, ns)| Entry::Choice { index, ns });
        match self.remember(Kind::Choice, key, within, || (programs(), ()), timed)? {
            Entry::Choice { index, .. } => Some(index),
            _ => None,
        }
    }

    /// A measured number (a probe's cost, say) by the same path as
    /// [`Self::select`]: the memo, the store, else `measure`, stored when `Some`.
    pub fn value(
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
    fn remember<T>(
        &self,
        kind: Kind,
        key: &TuneKey,
        valid: impl Fn(&Entry) -> bool,
        programs: impl FnOnce() -> (u64, T),
        measure: impl FnOnce(T) -> Option<Entry>,
    ) -> Option<Entry> {
        once(&self.memo, &(kind, key.clone()), || {
            let (programs, built) = programs();
            let line = key.line(kind, programs);
            self.entries(key).remove(&line).filter(&valid).or_else(|| {
                let entry = measure(built)?;
                self.put(key, line, &entry);
                Some(entry)
            })
        })
    }

    /// [`Self::select`] among `candidates`, keyed by the fingerprint of the
    /// programs `build` makes (each built once); only a store miss compiles
    /// them (those `bench` has not) and [`time`]s them.
    pub fn choose<C: Copy + PartialEq>(
        &self,
        key: &TuneKey,
        candidates: &[C],
        build: impl Fn(C) -> Candidate,
        bench: &mut Bench<C>,
    ) -> Option<usize> {
        let programs = || {
            let built: Vec<Candidate> = candidates.iter().map(|&c| build(c)).collect();
            (fingerprint(&built), built)
        };
        let within = |entry: &Entry| matches!(entry, Entry::Choice { index, .. } if *index < candidates.len());
        let timed = |built| fastest(bench.time(candidates, built)).map(|(index, ns)| Entry::Choice { index, ns });
        match self.remember(Kind::Choice, key, within, programs, timed)? {
            Entry::Choice { index, .. } => Some(index),
            _ => None,
        }
    }

    /// The indices `admit` finds for the ranked list `key` names (`None`:
    /// nothing compiled to judge) by the same path as [`Self::select`]: the
    /// memo, the store under the line `programs` fingerprints, else `admit`,
    /// stored when it judged. What compiles without spilling is the
    /// compiler's and the programs' alone, both in the line, so a later
    /// process compiles nothing but its choice.
    pub fn admitted(
        &self,
        key: &TuneKey,
        programs: impl FnOnce() -> u64,
        admit: impl FnOnce() -> Option<Vec<usize>>,
    ) -> Option<Arc<[usize]>> {
        let judged = |()| admit().map(|at| Entry::Admitted(at.into()));
        match self.remember(Kind::Admitted, key, |_| true, || (programs(), ()), judged)? {
            Entry::Admitted(at) => Some(at),
            _ => None,
        }
    }
}

/// The fastest of candidates' device times as `(index, ns)`.
fn fastest(ns: Vec<Option<u64>>) -> Option<(usize, u64)> {
    ns.into_iter().enumerate().filter_map(|(i, ns)| Some((ns?, i))).min().map(|(ns, i)| (i, ns))
}

/// A candidate compiled on scratch buffers: its plans (`None` when one
/// failed to build) and its kernels' compiled resources where the backend
/// reports them.
pub struct Built {
    plans: Option<Vec<Scratch>>,
    resources: Vec<KernelResources>,
}

/// What compiling a candidate showed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compiled {
    Failed,
    /// A kernel [`spilled`]: timed only where every candidate does.
    Spilled,
    Clean,
}

impl Built {
    pub fn verdict(&self) -> Compiled {
        match &self.plans {
            None => Compiled::Failed,
            Some(_) if spilled(&self.resources) => Compiled::Spilled,
            Some(_) => Compiled::Clean,
        }
    }
}

/// Whether a kernel of `resources` spilled to scratch; a backend reporting
/// nothing has not.
pub fn spilled(resources: &[KernelResources]) -> bool {
    resources.iter().any(|r| r.scratch_bytes.is_some_and(|b| b > 0))
}

/// Candidates compiled once on one scratch pool and kept: admitted in ranked
/// batches, then timed by the passes that ask for them, none compiled twice.
pub struct Bench<C> {
    pool: ScratchPool,
    pub(crate) built: Vec<(C, Built)>,
}

impl<C> Default for Bench<C> {
    fn default() -> Self {
        Self { pool: ScratchPool::default(), built: vec![] }
    }
}

impl<C: Copy + PartialEq> Bench<C> {
    fn find(&self, c: C) -> Option<usize> {
        self.built.iter().position(|(b, _)| *b == c)
    }

    /// Compiles, in parallel, those of `candidates` not compiled before,
    /// from the programs `programs` makes of a candidate's position.
    fn compile(&mut self, candidates: &[C], mut programs: impl FnMut(usize) -> Candidate) {
        let missing: Vec<usize> = (0..candidates.len()).filter(|&i| self.find(candidates[i]).is_none()).collect();
        let built = compile(missing.iter().map(|&i| programs(i)).collect(), &self.pool);
        self.built.extend(missing.into_iter().map(|i| candidates[i]).zip(built));
    }

    /// What compiling each of `batch` (the programs `build` makes) showed.
    pub fn admit(&mut self, batch: &[C], build: impl Fn(C) -> Candidate) -> Vec<Compiled> {
        self.compile(batch, |i| build(batch[i]));
        batch.iter().map(|&c| self.built[self.find(c).expect("compiled")].1.verdict()).collect()
    }

    /// Each of `candidates`' best device time ([`time`]); those not compiled
    /// yet compile from `programs`, theirs in order.
    pub fn time(&mut self, candidates: &[C], programs: Vec<Candidate>) -> Vec<Option<u64>> {
        let mut programs: Vec<Option<Candidate>> = programs.into_iter().map(Some).collect();
        self.compile(candidates, |i| programs[i].take().expect("a candidate's programs"));
        let (timed, mut built): (Vec<C>, Vec<&mut Built>) =
            self.built.iter_mut().filter(|(c, _)| candidates.contains(c)).map(|(c, b)| (*c, b)).unzip();
        let ns = time(&mut built);
        candidates.iter().map(|c| ns[timed.iter().position(|t| t == c).expect("compiled")]).collect()
    }
}

/// Each candidate's best device time in ns, its programs' summed, on scratch
/// buffers at capacity: [`compile`], then [`time`].
pub fn measure(candidates: impl IntoIterator<Item = Candidate>) -> Vec<Option<u64>> {
    time(&mut compile(candidates.into_iter().collect(), &ScratchPool::default()))
}

/// Builds each candidate's plans on `pool`'s scratch buffers at capacity
/// (every runtime variable bound to its maximum) and reads its kernels'
/// [`resources`]. The programs compile in parallel, on workers carrying the
/// caller's default device and origin scope.
pub(crate) fn compile(candidates: Vec<Candidate>, pool: &ScratchPool) -> Vec<Built> {
    in_parallel(candidates, |programs| {
        let plans: Option<Vec<Scratch>> = programs.into_iter().map(|(p, l)| scratch_plan(p, &l, pool)).collect();
        let resources = plans.iter().flatten().flat_map(|(plan, _)| resources(plan)).collect();
        Built { plans, resources }
    })
}

/// Held across every [`time`] in the process, the probes' and the tuner's
/// alike: two measurements sharing the device would time each other.
static DEVICE: Mutex<()> = Mutex::new(());

/// Each built candidate's best device time in ns, its programs' summed;
/// `None` where one failed to build or run. The device runs them one at a
/// time, after the clock is lifted, under the process's one timing lock.
pub(crate) fn time<B: BorrowMut<Built>>(built: &mut [B]) -> Vec<Option<u64>> {
    let _device = DEVICE.lock().unwrap_or_else(PoisonError::into_inner);
    let plans: Vec<Option<Vec<&ExecutionPlan>>> = built
        .iter_mut()
        .map(|b| {
            b.borrow_mut()
                .plans
                .as_mut()?
                .iter_mut()
                .map(|(plan, vars)| {
                    let vars: Vec<(&str, i64)> = vars.iter().map(|(n, v)| (n.as_str(), *v)).collect();
                    plan.execute_with_vars(&vars).ok().map(|_| &*plan)
                })
                .collect()
        })
        .collect();
    if let Some(first) = plans.iter().flatten().flatten().next() {
        warm_clock(CLOCK_WARMUP, || run(first));
    }
    // A plan's last timed run sizes its sustain, one timed run first.
    let mut last: Vec<Vec<Option<Duration>>> =
        plans.iter().map(|p| vec![None; p.as_ref().map_or(0, Vec::len)]).collect();
    let time = |i: usize| {
        let plans = plans[i].as_ref()?;
        plans
            .iter()
            .zip(&mut last[i])
            .map(|(plan, last)| {
                let per_run = match *last {
                    Some(t) => t,
                    None => run(plan)?,
                };
                sustain(plan, SUSTAIN, per_run);
                let t = best((0..RUNS).map(|_| run(plan)))?;
                *last = Some(t);
                Some(t)
            })
            .sum::<Option<Duration>>()
    };
    rounds(plans.len(), time).into_iter().map(|t| t.map(|t| t.as_nanos() as u64)).collect()
}

/// Each candidate's minimum over the timed rounds: one round of every
/// candidate, then [`ROUNDS`]` - 1` more ([`round_robin_min`]) of the
/// [`CONTENDER`]s of the first round's best. `None` where a candidate
/// failed to run.
pub(crate) fn rounds(count: usize, mut time: impl FnMut(usize) -> Option<Duration>) -> Vec<Option<Duration>> {
    let mut best = round_robin_min(count, 1, None, &mut time);
    let Some(lead) = best.iter().flatten().min().copied() else { return best };
    let bound = lead.mul_f64(CONTENDER);
    let contenders: Vec<usize> = (0..count).filter(|&i| best[i].is_some_and(|t| t <= bound)).collect();
    let more = round_robin_min(contenders.len(), ROUNDS - 1, Some(bound), |j| time(contenders[j]));
    for (&i, more) in contenders.iter().zip(more) {
        best[i] = more.and_then(|m| best[i].map(|b| b.min(m)));
    }
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
                        // every plan that reads it, one per candidate.
                        let t = Tensor::zeros(&[p.elems], p.dtype.into()).contiguous();
                        t.realize().expect("a scratch buffer");
                        t
                    })
                    .clone()
            })
            .collect()
    }
}

/// A candidate's plan on `pool`'s buffers: lowered and prepared for this
/// measurement alone, kept in no memo or cache (it runs under scratch
/// placeholders no op's call keys, and a shortlist of them per shape would
/// otherwise stay for the process).
pub(crate) fn scratch_plan(prog: Program, lowering: &Lowering, pool: &ScratchPool) -> Option<Scratch> {
    let vars: Vec<(String, i64)> = prog.vars.iter().map(|v| (v.name.clone(), v.max)).collect();
    let buffers = pool.buffers(&prog.params);
    let out = launch::graph_launch_once(prog, lowering, &buffers.iter().collect::<Vec<_>>()).ok()?;
    let mut config = PrepareConfig::from_env();
    config.ephemeral = true;
    Some((out.prepare_with(&config).ok()?, vars))
}

/// Runs `plan` back to back, unsynchronized, for about `budget` of device
/// time at `per_run` a run. Launches queue without waiting, so a spin of
/// host time would queue a thousand runs of a short kernel and every timed
/// run after it would wait for the queue to drain.
fn sustain(plan: &ExecutionPlan, budget: Duration, per_run: Duration) {
    let launches = budget.as_nanos().div_ceil(per_run.as_nanos().max(1)).clamp(1, 1024);
    for _ in 0..launches {
        if plan.execute().is_err() {
            break;
        }
    }
}

/// `f` over `items` on the rayon pool, each worker running under the
/// caller's default device and origin scope (both per thread), so what it
/// builds is what the caller would have built.
pub(crate) fn in_parallel<T: Send, R: Send>(items: Vec<T>, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    use rayon::prelude::*;
    let (device, origin, capture) = (default_device(), svod_ir::origin::current(), svod_ir::origin::enabled());
    items
        .into_par_iter()
        .map(|item| {
            with_default_device(device.clone(), || {
                let _capture = svod_ir::origin::capture_for_thread(capture);
                let _scope = svod_ir::origin::install(origin);
                f(item)
            })
        })
        .collect()
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
pub fn resources(plan: &ExecutionPlan) -> Vec<KernelResources> {
    plan.kernels().filter_map(|k| k.program.resource_usage()).collect()
}

/// `0`: follow the environment; `1`: off; `2`: on.
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// [`set_enabled`]'s last setting, else `SVOD_TK3_TUNE` is not `0`.
pub fn enabled() -> bool {
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => std::env::var("SVOD_TK3_TUNE").map_or(true, |v| v != "0"),
    }
}

/// Force measuring on or off for this process, over the environment.
pub fn set_enabled(on: bool) {
    OVERRIDE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
}
