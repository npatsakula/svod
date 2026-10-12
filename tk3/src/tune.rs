//! Measured config choice per shape. An op's candidate list is a search space:
//! the first time a device meets a shape, every candidate is built on scratch
//! buffers at capacity and timed (the clock lifted first, then round-robin
//! rounds keeping each candidate's minimum), and the winner is kept in the
//! process memo and on disk. This runs where the op is called, at graph
//! build, never inside a running plan.
//!
//! The store is one line per entry (`key index ns`) in `$SVOD_TK3_TUNE_DIR`
//! (else `$XDG_CACHE_HOME/svod/tk3_tune`, else `$HOME/.cache/svod/tk3_tune`),
//! one file per device and crate version. A line's key carries a fingerprint
//! of the candidate programs, so a kernel change re-measures. An unreadable
//! or unwritable store is a miss, never an error; `SVOD_TK3_TUNE=0` or
//! [`set_enabled`] turns measuring off and the list's first entry is used.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use svod_dtype::ScalarDType;
use svod_dtype::default_device::{default_device, with_default_device};
use svod_runtime::ExecutionPlan;
use svod_runtime::benchmark::round_robin_min;
use svod_tensor::Tensor;

use crate::atoms::Target;
use crate::ir::{Param, Program};
use crate::launch;
use crate::lower::Lowering;

/// Timed rounds over every candidate after the warm-up.
const ROUNDS: usize = 4;
/// Profiled runs per candidate per round.
const RUNS: usize = 5;
/// Back-to-back runs lift a cold clock (an RTX 3060 idles at 210 MHz) ...
const WARMUP: Duration = Duration::from_millis(500);
/// ... and keep it up ahead of each candidate's profiled runs, whose host
/// overhead would otherwise let a short kernel's clock sag.
const SUSTAIN: Duration = Duration::from_millis(10);

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
}

impl TuneKey {
    pub fn new(op: &'static str, target: &Target, dtype: ScalarDType, shape: &[usize], candidates: &impl Hash) -> Self {
        let device = format!("{}-{}sm", target.arch.target_name(), target.sms.unwrap_or(0));
        Self { op, device, dtype, shape: shape.to_vec(), candidates: digest(candidates) }
    }

    /// The store line for the candidate programs fingerprinted as `programs`.
    pub fn line(&self, programs: u64) -> String {
        let shape: Vec<String> = self.shape.iter().map(usize::to_string).collect();
        let (op, device, dtype) = (self.op, &self.device, self.dtype);
        format!("{op}|{device}|{dtype:?}|{}|{:016x}|{programs:016x}", shape.join("x"), self.candidates)
    }
}

/// A candidate's programs, launched in order (a kernel and the merge of
/// its partial results, say).
pub type Candidate = Vec<(Program, Lowering)>;

/// The fingerprint of built candidates: their tile programs and lowerings.
pub fn fingerprint(candidates: &[Candidate]) -> u64 {
    digest(&format!("{candidates:?}"))
}

/// A memo of this process's choices, backed by one file per device under `root`.
#[derive(Debug)]
pub struct TuneStore {
    root: Option<PathBuf>,
    memo: Mutex<HashMap<TuneKey, Option<usize>>>,
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

    /// Every `line -> (index, ns)` of the key's device file; empty when unreadable.
    pub fn entries(&self, key: &TuneKey) -> HashMap<String, (usize, u64)> {
        let Some(text) = self.path(key).and_then(|p| std::fs::read_to_string(p).ok()) else { return HashMap::new() };
        text.lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let (line, index) = (fields.next()?.to_string(), fields.next()?.parse().ok()?);
                Some((line, (index, fields.next()?.parse().ok()?)))
            })
            .collect()
    }

    /// Re-read, merge and replace the file atomically, so concurrent writers
    /// lose at most each other's newest line, never the file.
    fn put(&self, key: &TuneKey, line: String, index: usize, ns: u64) {
        let Some(path) = self.path(key) else { return };
        let mut entries = self.entries(key);
        entries.insert(line, (index, ns));
        let mut lines: Vec<String> = entries.iter().map(|(line, (i, ns))| format!("{line} {i} {ns}")).collect();
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
        if let Some(chosen) = self.memo.lock().expect("tune memo").get(key) {
            return *chosen;
        }
        let line = key.line(programs());
        let stored = self.entries(key).remove(&line).map(|(i, _)| i).filter(|i| *i < count);
        let chosen = stored.or_else(|| {
            let (ns, i) = measure().into_iter().enumerate().filter_map(|(i, ns)| Some((ns?, i))).min()?;
            self.put(key, line, i, ns);
            Some(i)
        });
        self.memo.lock().expect("tune memo").insert(key.clone(), chosen);
        chosen
    }

    /// [`Self::select`] over the programs `build(i)` makes for candidate `i`,
    /// built only on a memo miss and timed by [`measure`] on a store miss.
    pub fn choose(&self, key: &TuneKey, count: usize, build: impl Fn(usize) -> Candidate) -> Option<usize> {
        let programs = OnceLock::new();
        let built = || programs.get_or_init(|| (0..count).map(&build).collect::<Vec<_>>());
        self.select(key, count, || fingerprint(built()), || measure(built().iter().cloned()))
    }

    /// The candidate [`Self::choose`] finds among `candidates`, else the first.
    pub fn pick<C: Copy>(&self, key: &TuneKey, candidates: &[C], build: impl Fn(C) -> Candidate) -> C {
        let chosen = self.choose(key, candidates.len(), |i| build(candidates[i]));
        candidates[chosen.unwrap_or(0)]
    }
}

/// Each candidate's best device time in ns, its programs' summed, on scratch
/// buffers at capacity (every runtime variable bound to its maximum); `None`
/// where one fails. The programs compile in parallel, on workers carrying
/// the caller's default device and origin scope; the device runs them one
/// at a time.
pub fn measure(candidates: impl IntoIterator<Item = Candidate>) -> Vec<Option<u64>> {
    let candidates: Vec<Candidate> = candidates.into_iter().collect();
    let pool = ScratchPool::default();
    let compiled: Vec<Option<Vec<Scratch>>> =
        in_parallel(candidates, |programs| programs.into_iter().map(|(p, l)| scratch_plan(p, &l, &pool)).collect());
    let plans: Vec<Option<Vec<ExecutionPlan>>> = compiled
        .into_iter()
        .map(|plans| {
            plans?
                .into_iter()
                .map(|(mut plan, vars)| {
                    let vars: Vec<(&str, i64)> = vars.iter().map(|(n, v)| (n.as_str(), *v)).collect();
                    plan.execute_with_vars(&vars).ok().map(|_| plan)
                })
                .collect()
        })
        .collect();
    if let Some(first) = plans.iter().flatten().flatten().next() {
        spin(first, WARMUP);
    }
    let time = |i: usize| {
        let plans = plans[i].as_ref()?;
        plans
            .iter()
            .map(|plan| {
                spin(plan, SUSTAIN);
                best((0..RUNS).map(|_| run(plan)))
            })
            .sum::<Option<Duration>>()
    };
    round_robin_min(plans.len(), ROUNDS, None, time).into_iter().map(|t| t.map(|t| t.as_nanos() as u64)).collect()
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

fn scratch_plan(prog: Program, lowering: &Lowering, pool: &ScratchPool) -> Option<Scratch> {
    let vars: Vec<(String, i64)> = prog.vars.iter().map(|v| (v.name.clone(), v.max)).collect();
    let buffers = pool.buffers(&prog.params);
    let out = launch::graph_launch(prog, lowering, &buffers.iter().collect::<Vec<_>>()).ok()?;
    Some((out.prepare().ok()?, vars))
}

fn spin(plan: &ExecutionPlan, budget: Duration) {
    let start = std::time::Instant::now();
    while start.elapsed() < budget && plan.execute().is_ok() {}
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

/// The fastest of the runs that were timed; `None` only when none was.
pub(crate) fn best(runs: impl Iterator<Item = Option<Duration>>) -> Option<Duration> {
    runs.flatten().min()
}

/// One run's longest kernel; `None` when the device stamps no times.
fn run(plan: &ExecutionPlan) -> Option<Duration> {
    let profile = plan.execute_profiled().ok()?;
    profile.iter().filter_map(|k| k.gpu_end_ns?.checked_sub(k.gpu_start_ns?)).max().map(Duration::from_nanos)
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
