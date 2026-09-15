//! First-use tuning of the hand kernels' tile tables.
//!
//! A kernel's per-family table is a search space, not an answer: the first time
//! a device meets a shape, every candidate that fits it is compiled and timed
//! on synthetic operands of that shape — the clock lifted first, for as long as
//! the kernel's time keeps falling ([`svod_runtime::benchmark::warm_clock`]; a
//! device already under load pays a few dozen runs), the candidates timed in turn over
//! several rounds ([`svod_runtime::benchmark::round_robin_min`]) so none is
//! judged at a clock the others were not — and the winner is kept, in the
//! store's memo and on disk so the next process starts tuned. Measurement is
//! skipped, and the table's static choice used, when tuning is off
//! (`SVOD_TK_TUNE=0`, or [`set_enabled`], which the test harnesses use), when
//! the device stamps no timings, or when no candidate runs; nothing unmeasured
//! is ever cached.
//!
//! The store is one line per entry (`key index ns`) in `$SVOD_TK_TUNE_DIR`
//! (else `$XDG_CACHE_HOME/svod/tk_tune`, else `$HOME/.cache/svod/tk_tune`), one
//! file per device and crate version. The key carries a fingerprint of the
//! candidate kernels' graphs, so a kernel change re-measures; an unreadable or
//! unwritable store is a miss, never an error.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use svod_dtype::{DeviceSpec, GpuArch};
use svod_runtime::benchmark::{CLOCK_WARMUP, round_robin_min, warm_clock};

use crate::launch::CompiledLaunch;

/// Timed rounds over the candidates after the warm-up.
const ROUNDS: usize = 3;

/// What a measurement is keyed by: the device (arch and compute units), the
/// kernel, its shape, and the candidate kernels themselves (their graph
/// fingerprints, so a table or kernel change re-measures).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TuneKey {
    pub kernel: &'static str,
    pub device: String,
    pub shape: Vec<usize>,
    pub builds: u64,
}

impl TuneKey {
    /// The key for `kernel` at `shape` on the device behind `spec`, over the
    /// candidates whose built graphs have the [`crate::kernel_fingerprint`]
    /// digests `builds` (in table order).
    pub fn new(kernel: &'static str, spec: &DeviceSpec, arch: GpuArch, shape: &[usize], builds: &[u128]) -> Self {
        let mut hasher = std::hash::DefaultHasher::new();
        builds.hash(&mut hasher);
        let units = crate::target::compute_units(spec).unwrap_or(0);
        Self {
            kernel,
            device: format!("{}-{units}cu", arch.target_name()),
            shape: shape.to_vec(),
            builds: hasher.finish(),
        }
    }

    fn line(&self) -> String {
        let shape: Vec<String> = self.shape.iter().map(usize::to_string).collect();
        format!("{}|{}|{}|{:016x}", self.kernel, self.device, shape.join("x"), self.builds)
    }
}

/// The store: a memo of this process's choices, backed by one file per device
/// under `root` when there is one.
#[derive(Debug)]
pub struct TuneStore {
    root: Option<PathBuf>,
    memo: Mutex<HashMap<TuneKey, usize>>,
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
            let root = if let Some(dir) = std::env::var_os("SVOD_TK_TUNE_DIR") {
                Some(PathBuf::from(dir))
            } else if let Some(cache) = std::env::var_os("XDG_CACHE_HOME") {
                Some(PathBuf::from(cache).join("svod/tk_tune"))
            } else {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/svod/tk_tune"))
            };
            Self::at(root.filter(|dir| std::fs::create_dir_all(dir).is_ok()))
        })
    }

    fn path(&self, key: &TuneKey) -> Option<PathBuf> {
        let name: String = key.device.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        self.root.as_ref().map(|root| root.join(format!("{name}-v{}.txt", env!("CARGO_PKG_VERSION"))))
    }

    /// Every `key line -> (index, ns)` the device's file holds; empty when unreadable.
    fn read(&self, key: &TuneKey) -> HashMap<String, (usize, u64)> {
        let Some(text) = self.path(key).and_then(|p| std::fs::read_to_string(p).ok()) else { return HashMap::new() };
        text.lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let (line, index) = (fields.next()?.to_string(), fields.next()?.parse().ok()?);
                Some((line, (index, fields.next()?.parse().ok()?)))
            })
            .collect()
    }

    fn get(&self, key: &TuneKey) -> Option<usize> {
        self.read(key).remove(&key.line()).map(|(index, _)| index)
    }

    /// Record `index` (measured at `ns`) for `key`: re-read, merge, and replace
    /// the file atomically, so concurrent writers lose at most each other's
    /// newest line, never the file.
    fn put(&self, key: &TuneKey, index: usize, ns: u64) {
        let Some(path) = self.path(key) else { return };
        let mut entries = self.read(key);
        entries.insert(key.line(), (index, ns));
        let mut lines: Vec<String> = entries.iter().map(|(line, (i, ns))| format!("{line} {i} {ns}")).collect();
        lines.sort();
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        if std::fs::write(&tmp, lines.join("\n") + "\n").is_ok() && std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }

    /// The winning candidate index for `key` among `count` candidates: the
    /// memo, then the store, else `measure` times them all (each candidate's
    /// device ns, `None` where it cannot run) and the fastest is kept. `None`
    /// when nothing measured — the caller keeps its static choice.
    pub fn select_with(
        &self,
        key: &TuneKey,
        count: usize,
        measure: impl FnOnce() -> Vec<Option<u64>>,
    ) -> Option<usize> {
        if let Some(i) = self.memo.lock().expect("tune memo").get(key) {
            return Some(*i);
        }
        let cached = self.get(key).filter(|i| *i < count);
        let chosen = cached.or_else(|| {
            let (ns, i) = measure().into_iter().enumerate().filter_map(|(i, ns)| ns.map(|ns| (ns, i))).min()?;
            self.put(key, i, ns);
            Some(i)
        })?;
        self.memo.lock().expect("tune memo").insert(key.clone(), chosen);
        Some(chosen)
    }

    /// [`Self::select_with`] over kernels: `compile(i)` builds candidate `i` (or
    /// `None` when it cannot be built); the first that built lifts the clock,
    /// then every candidate is timed in turn for [`ROUNDS`] rounds and its
    /// minimum kept.
    pub fn select(
        &self,
        key: &TuneKey,
        count: usize,
        compile: impl FnMut(usize) -> Option<CompiledLaunch>,
    ) -> Option<usize> {
        self.select_with(key, count, || {
            let launches: Vec<Option<CompiledLaunch>> = (0..count).map(compile).collect();
            if let Some(first) = launches.iter().flatten().next() {
                // SAFETY: the launch's buffers live in `first` for the whole loop.
                warm_clock(CLOCK_WARMUP, || first.dispatch_gpu_ns().ok().flatten().map(Duration::from_nanos));
            }
            let time = |i: usize| launches[i].as_ref()?.dispatch_gpu_ns().ok().flatten().map(Duration::from_nanos);
            round_robin_min(count, ROUNDS, time).into_iter().map(|t| t.map(|t| t.as_nanos() as u64)).collect()
        })
    }
}

/// `0`: follow the environment; `1`: off; `2`: on.
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Whether first-use measurement is on: [`set_enabled`]'s last setting, else
/// `SVOD_TK_TUNE` is not `0`.
pub fn enabled() -> bool {
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => std::env::var("SVOD_TK_TUNE").map(|v| v != "0").unwrap_or(true),
    }
}

/// Force measurement on or off for this process, over the environment — the
/// test harnesses turn it off so a kernel test does not tune every shape it
/// touches.
pub fn set_enabled(on: bool) {
    OVERRIDE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
}
