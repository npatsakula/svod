//! The rank harness: a tune store replayed offline, correlating the order
//! the model predicts a shape's timed candidates with the ns the device
//! measured for them. The instrument that gates every later model change
//! (the config rework's §7.1): a store's `choice|` line holds every timed
//! candidate of a shape, fastest first, each with its config's own code
//! beside its index — replay prices the line's own candidates, so a store
//! reads whatever ranking code wrote it and whatever ranks since (a model
//! change no longer orphans a measurement). The digest in the line's key
//! stays the consistency check: where the ranking code at hand re-derives
//! the very list a line timed, the line counts as derived; the rest replay
//! from their codes under the problem of the hypothesis whose ranking
//! admits the most of them, and a line no hypothesis admits — an unknown
//! salt — is skipped and counted, never guessed at.
//!
//! Per family and device: the Spearman ρ of predicted against measured
//! order over the replayed lines, and the fraction of lines whose predicted
//! best is the measured winner — the number tuning actually cares about.
//! The derived stage follows [`crate::ops::Tuner`]: a shape's ranking
//! admitted, its candidates expanded as the op layer expands them,
//! screened, and the leading tiles' pipelines timed again; a stage's lookup
//! missing (a wrong salt) hands the shape to the next hypothesis.

use std::collections::HashMap;
use std::hash::Hasher;
use std::path::Path;

use svod_dtype::{AmdArch, CudaArch, GpuArch, ScalarDType};

use crate::atoms::{CARDS, Target};
use crate::ir::Var;
use crate::kernels::attention::{AttnMask, Bias, FaCfg};
use crate::kernels::conv::{ConvCfg, ConvGeom};
use crate::kernels::gemm::{Epilogue, GemmCfg};
use crate::kernels::nearest::{Floors, Live, Metric, NearestCfg, NearestSpec};
use crate::kernels::{Act, Batch};
use crate::ops::DEVICE_TEST;
use crate::ops::attention::split_ranked;
use crate::ops::config::{
    self, AttentionProblem, ConvProblem, GemmProblem, NearestProblem, Planner, Tiled, representatives,
};
use crate::ops::nearest::Salt;
use crate::tune::{self, Entry, Kind, TuneKey};

/// A choice line the harness correlates holds at least this many timed
/// candidates: a pair of softmax forms is a flip the model does not price
/// (its configs tie), a line with no order to check.
const LEAST_CANDIDATES: usize = 3;

/// Whether `key` is a line of `group` exactly: the group's fields whole,
/// not a shape whose joined text begins another's (`128` of `128x16`).
fn of_group(key: &str, group: &str) -> bool {
    key.len() > group.len() && key.as_bytes()[group.len()] == b'|' && &key[..group.len()] == group
}

/// The choice-line group of `op`'s `dtype` and `shape` on `planner`'s
/// device: every line of the shape, any salt ([`TuneKey`]'s key up to the
/// candidates' digest).
fn group(op: &str, planner: &Planner, dtype: ScalarDType, shape: &[usize]) -> String {
    let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
    format!("choice|{op}|{}|{dtype:?}|{}", crate::tune::device(&planner.device.target), shape.join("x"))
}

fn digest(value: &impl std::hash::Hash) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// A choice line's candidates: index into the list its digest names, the
/// config's code, and its measured ns.
type Timed = Vec<(usize, String, u64)>;

/// One device's store lines, by key.
struct Lines {
    /// Line key (up to the candidates' digest; whatever compiler and
    /// programs wrote it) to its entry, several builds' lines of one key
    /// holding the one the file's order gives.
    entries: HashMap<String, Entry>,
    /// The lines a derived list's digest matched (every line of the key's
    /// prefix, builds alike): what the consistency check confirmed.
    read: std::collections::HashSet<String>,
    /// The payloads of the lines replayed from their own candidates (no
    /// derivation matching their digest): the cross-ranked measurements.
    stored: std::collections::HashSet<u64>,
}

impl Lines {
    /// Every store file under `root` of this crate's version, its lines
    /// merged by the device they name.
    fn read(root: &Path) -> HashMap<String, Self> {
        let mut out: HashMap<String, Self> = HashMap::new();
        let suffix = format!("-v{}.txt", env!("CARGO_PKG_VERSION"));
        for path in std::fs::read_dir(root).into_iter().flatten().flatten().map(|e| e.path()) {
            if !path.is_file() || !path.file_name().is_some_and(|n| n.to_string_lossy().ends_with(&suffix)) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            if text.lines().next() != Some(format!("format {}", tune::FORMAT).as_str()) {
                continue;
            }
            let entries = Self::parse(&text);
            let Some(key) = entries.keys().next() else { continue };
            let device = key.split('|').nth(2).unwrap_or_default().to_string();
            let lines = out.entry(device).or_insert_with(|| Self {
                entries: HashMap::new(),
                read: Default::default(),
                stored: Default::default(),
            });
            lines.entries.extend(entries);
        }
        out
    }

    /// Every readable `line -> entry` of a store file's text.
    fn parse(text: &str) -> HashMap<String, Entry> {
        text.lines()
            .skip(1)
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let key = fields.next()?;
                let numbers: Vec<&str> = fields.collect();
                Some((key.to_string(), Entry::parse(key.split('|').next()?, &numbers)?))
            })
            .collect()
    }

    /// The entry of `kind` a stage's `list` keys under `keying`, `None` when
    /// no line holds that key: the digest asserts the list is the one the
    /// store timed.
    fn at<C: std::hash::Hash + std::fmt::Debug>(
        &mut self,
        kind: Kind,
        keying: &Keying,
        rule: &str,
        list: &[C],
    ) -> Option<Entry> {
        let candidates = format!("{}{rule} {list:?}", keying.salt);
        let key = TuneKey::new(keying.op, &keying.target, 0, keying.dtype, &keying.shape, &candidates);
        let prefix = key.candidates(kind);
        let mut found: Vec<&String> = self.entries.keys().filter(|l| l.starts_with(&prefix)).collect();
        found.sort();
        let entry = found.first().map(|l| self.entries[*l].clone());
        if entry.is_some() {
            self.read.extend(found.into_iter().cloned());
        }
        entry
    }

    /// The candidates that ran, fastest first, each with its ns.
    fn choice<C: std::hash::Hash + std::fmt::Debug>(
        &mut self,
        keying: &Keying,
        rule: &str,
        list: &[C],
    ) -> Option<Vec<(usize, u64)>> {
        match self.at(Kind::Choice, keying, rule, list)? {
            Entry::Choice(timed) => Some(timed.iter().map(|&(i, _, ns)| (i, ns)).collect()),
            _ => None,
        }
    }

    /// The ranked indices an admission kept.
    fn admitted<C: std::hash::Hash + std::fmt::Debug>(
        &mut self,
        keying: &Keying,
        rule: &str,
        list: &[C],
    ) -> Option<Vec<usize>> {
        match self.at(Kind::Admitted, keying, rule, list)? {
            Entry::Admitted(at) => Some(at.iter().map(|&(i, _)| i).collect()),
            _ => None,
        }
    }

    /// The index of the first clean config of a ranked list.
    fn first<C: std::hash::Hash + std::fmt::Debug>(&mut self, keying: &Keying, list: &[C]) -> Option<usize> {
        match self.at(Kind::First, keying, "", list)? {
            Entry::Admitted(at) => at.first().map(|&(i, _)| i),
            _ => None,
        }
    }

    /// The choice lines of `group` no derived list matched, each with its
    /// decoded candidates and measured ns, one of each distinct payload
    /// (builds write the same measurement on several lines).
    fn unread(&self, group: &str) -> Vec<(Timed, u64)> {
        let mut out: Vec<(Timed, u64)> = vec![];
        for (key, entry) in &self.entries {
            if !of_group(key, group) || self.read.contains(key) {
                continue;
            }
            let Entry::Choice(timed) = entry else { continue };
            let payload = digest(&timed);
            if out.iter().all(|(_, p)| *p != payload) {
                out.push((timed.to_vec(), payload));
            }
        }
        out
    }
}

/// What one tuning attempt keys every stage's lines by: the op, the
/// device's target, dtype, shape and salt.
struct Keying {
    op: &'static str,
    target: Target,
    dtype: ScalarDType,
    shape: Vec<usize>,
    salt: String,
}

impl Keying {
    fn new(op: &'static str, planner: &Planner, dtype: ScalarDType, shape: &[usize], salt: String) -> Self {
        Self { op, target: planner.device.target.clone(), dtype, shape: shape.to_vec(), salt }
    }
}

/// One replayed `choice|` line: the candidates its key asserts, what the
/// model says they cost, and the ns the device measured.
pub struct Replay {
    /// `" screened"` (the shortlist's first pass) or `""` (the leading
    /// tiles' pipelines timed again): the two stages of a shape's choice.
    pub rule: &'static str,
    pub configs: Configs,
    /// The candidates that ran, in the line's (fastest first) order.
    pub timed: Vec<(usize, u64)>,
}

/// The candidates a replayed line names, with the problem that ranks them.
pub enum Configs {
    Gemm {
        problem: GemmProblem,
        list: Vec<GemmCfg>,
    },
    Attention {
        problem: AttentionProblem,
        list: Vec<FaCfg>,
    },
    Nearest {
        problem: NearestProblem,
        list: Vec<NearestCfg>,
    },
    /// A convolution's GEMM body under its own claim (the image the fill's
    /// working set); its splits the model does not price (the ranking
    /// orders the unsplit configs alone).
    Conv {
        problem: ConvProblem,
        list: Vec<ConvCfg>,
    },
}

impl Configs {
    /// The ns of SM time the model ranks `at` of the list at.
    fn cost(&self, planner: &Planner, at: usize) -> Option<f64> {
        let of = |problem: &GemmProblem, cfg: &_| problem.admit(planner, cfg).ok().map(|(ns, _)| ns);
        match self {
            Self::Gemm { problem, list } => of(problem, &list[at]),
            Self::Attention { problem, list } => problem.admit(planner, &list[at]).ok().map(|(ns, _)| ns),
            Self::Nearest { problem, list } => problem.admit(planner, &list[at]).ok().map(|(ns, _)| ns),
            Self::Conv { problem, list } => problem.claim(planner, &list[at].gemm).ok().map(|c| planner.price(&c)),
        }
    }
}

impl Replay {
    /// The model's predicted ns per timed candidate, in the line's order.
    pub fn predicted(&self, planner: &Planner) -> Vec<f64> {
        self.timed.iter().map(|&(at, _)| self.configs.cost(planner, at).unwrap_or(f64::MAX)).collect()
    }

    /// The measured ns per timed candidate, in the line's order.
    fn measured(&self) -> Vec<f64> {
        self.timed.iter().map(|&(_, ns)| ns as f64).collect()
    }

    /// Whether the model's best is the measured winner: the pick the tuner
    /// would have made without timing.
    pub fn top1(&self, planner: &Planner) -> bool {
        let predicted = self.predicted(planner);
        let best = predicted
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(at, _)| self.timed[at].0)
            .expect("a timed line holds a candidate");
        best == self.timed[0].0
    }

    /// The Spearman ρ of the model's order against the device's.
    pub fn rho(&self, planner: &Planner) -> Option<f64> {
        spearman(&self.predicted(planner), &self.measured())
    }
}

/// One family's outcome over a device's store.
pub struct Family {
    pub name: &'static str,
    /// The replayed choice lines ([`LEAST_CANDIDATES`] candidates at least).
    pub lines: Vec<Replay>,
    /// Every choice line of the family's ops in the store.
    pub total: usize,
    /// The choice lines a replay found, and of them the ones too short to
    /// correlate (a pair of softmax forms).
    pub replayed: usize,
    pub short: usize,
    /// The replayed lines read from their own candidates, no derivation
    /// matching their digest: the cross-ranked measurements.
    pub stored: usize,
    /// The choice lines no hypothesis admitted a candidate of.
    pub skipped: usize,
}

impl Family {
    fn new(name: &'static str) -> Self {
        Self { name, lines: vec![], total: 0, replayed: 0, short: 0, stored: 0, skipped: 0 }
    }

    /// The mean Spearman ρ over the lines, each shape's measurement weighing
    /// alike.
    pub fn rho(&self, planner: &Planner) -> Option<f64> {
        let rhos: Vec<f64> = self.lines.iter().filter_map(|l| l.rho(planner)).collect();
        (!rhos.is_empty()).then(|| rhos.iter().sum::<f64>() / rhos.len() as f64)
    }

    /// The fraction of lines whose predicted best is the measured winner.
    pub fn top1(&self, planner: &Planner) -> Option<f64> {
        (!self.lines.is_empty())
            .then(|| self.lines.iter().filter(|l| l.top1(planner)).count() as f64 / self.lines.len() as f64)
    }

    /// The timed candidates over the lines.
    pub fn candidates(&self) -> usize {
        self.lines.iter().map(|l| l.timed.len()).sum()
    }
}

/// One device's replay: its name, the cost fields its store could not
/// measure, the planner its costs make, each family's lines, and the
/// `(op dtype shape, lines)` groups no hypothesis reproduced.
pub struct Device {
    pub name: String,
    pub unmeasured: Vec<&'static str>,
    pub planner: std::sync::Arc<Planner>,
    pub families: Vec<Family>,
    pub skipped_groups: Vec<(String, usize)>,
}

/// One nearest family's per-cluster ablation: the ρ of the replayed lines
/// with one cost cluster zeroed, against the ρ with every cluster at its
/// full weight — the cluster whose removal moves ρ most is where the
/// model's error lives.
pub struct Ablated {
    pub family: &'static str,
    pub base: Option<f64>,
    /// `(cluster, ρ with the cluster zeroed)`.
    pub clusters: Vec<(&'static str, Option<f64>)>,
}

/// Zeroes each of the nearest epilogue's cost clusters
/// ([`crate::ops::config`'s `Weights`]) in turn over the device's replayed
/// nearest lines, re-predicting them under the same measured costs: the
/// candidate lists and their measured ns stay the store's own.
pub fn ablation(device: &Device) -> Vec<Ablated> {
    use crate::ops::config::Weights;
    type Cluster = (&'static str, fn(&mut Weights));
    let clusters: [Cluster; 4] = [
        ("pair", |w| w.pair = 0.0),
        ("deal", |w| w.deal = 0.0),
        ("merge", |w| w.merge = 0.0),
        ("scan", |w| w.scan = 0.0),
    ];
    device
        .families
        .iter()
        .filter(|f| f.name.starts_with("nearest"))
        .filter(|f| !f.lines.is_empty())
        .map(|family| Ablated {
            family: family.name,
            base: family.rho(&device.planner),
            clusters: clusters
                .iter()
                .map(|&(name, zero)| {
                    let mut planner = Planner::new(device.planner.device.clone()).expect("an SM count");
                    zero(&mut planner.weights);
                    (name, family.rho(&planner))
                })
                .collect(),
        })
        .collect()
}

/// The model's cost fields one at a time perturbed (×`factor`, the ladder
/// with it) over the device's replayed `family` lines: the ρ each
/// perturbation leaves says which term the family's order rides on — the
/// explain mode's attribution, priced on the lines the store itself timed.
pub fn explained(device: &Device, family: &str, factor: f64) -> Vec<(&'static str, Option<f64>)> {
    let mut probe_costs = device.planner.device.costs;
    let mut fields: Vec<&'static str> = probe_costs.fields().into_iter().map(|(f, _)| f).collect();
    fields.extend(["l2_byte"]);
    fields
        .into_iter()
        .map(|field| {
            let mut costs = device.planner.device.costs;
            let ladder = |costs: &mut crate::atoms::Costs| {
                if field == "l2_byte" {
                    costs.l2_byte *= factor;
                    for (_, rate) in costs.ladder.iter_mut() {
                        *rate = (*rate * factor).min(costs.global_byte);
                    }
                }
            };
            match field {
                "step" => costs.step *= factor,
                "step_i8" => costs.step_i8 *= factor,
                "score" => costs.score *= factor,
                "smem_byte" => costs.smem_byte *= factor,
                "global_byte" => {
                    costs.global_byte *= factor;
                    ladder(&mut costs);
                }
                "trip" => costs.trip *= factor,
                "trip_one" => costs.trip_one *= factor,
                "latency" => costs.latency *= factor,
                "smem_trip" => costs.smem_trip *= factor,
                "lane_move" => costs.lane_move *= factor,
                "l2_byte" => ladder(&mut costs),
                _ => return (field, None),
            }
            let planner =
                Planner::new(crate::atoms::Device { costs, ..device.planner.device.clone() }).expect("an SM count");
            let family = device.families.iter().find(|f| f.name == family);
            (field, family.and_then(|f| f.rho(&planner)))
        })
        .collect()
}

/// The layer-2 fit, evaluated: the multiplier a family's card earns for
/// the declared residual unit ([`crate::ops::config::pair_op`]'s 0.2 score
/// elements an elementary operation) from its own store's timed outcomes —
/// every ops kind's weight scaled together, the whole epilogue's unit. One
/// leave-one-out pass over the replayed lines: each line held out in turn,
/// the multiplier that orders the rest best read off a geometric grid
/// around the prior, and the held-out line predicted under it and under the
/// prior (the multiplier 1). The fit is accepted only where the mean
/// held-out ρ beats the prior's; enabling it anywhere is the design's
/// phase F, so this computes and reports, never plans.
pub struct Fitted {
    pub family: &'static str,
    /// The multiplier each held-out fold fitted, in its line's order.
    pub multipliers: Vec<f64>,
    /// The mean held-out ρ at the folds' fitted multipliers, against the
    /// prior's (every weight 1).
    pub held_out: Option<f64>,
    pub prior: Option<f64>,
    /// The multiplier that orders every line best, the one a store would
    /// keep (regularized: the grid point nearest 1 among those within a
    /// hundredth of the best ρ).
    pub overall: f64,
}

/// The grid the fit reads its multiplier off: a quarter to four times the
/// prior, the prior itself among the points.
const FIT_GRID: [f64; 13] = [0.25, 0.3, 0.36, 0.44, 0.53, 0.72, 1.0, 1.4, 1.7, 2.1, 2.5, 3.1, 4.0];

pub fn fitted(device: &Device) -> Vec<Fitted> {
    let scaled = |m: f64| {
        let mut planner = Planner::new(device.planner.device.clone()).expect("an SM count");
        planner.weights = crate::ops::config::Weights { pair: m, deal: m, merge: m, scan: m, epilogue: m };
        planner
    };
    let mean = |rho: &[Option<f64>]| {
        let kept: Vec<f64> = rho.iter().flatten().copied().collect();
        (!kept.is_empty()).then(|| kept.iter().sum::<f64>() / kept.len() as f64)
    };
    device
        .families
        .iter()
        .filter(|f| f.lines.len() >= 4)
        .map(|family| {
            // The multiplier that orders `keep`'s lines best: the best ρ
            // over the grid, regularized to the point nearest the prior
            // among those within five hundredths of it.
            let best_over = |keep: &[usize]| {
                let scored: Vec<(f64, f64)> = FIT_GRID
                    .iter()
                    .map(|&m| {
                        let rho: Vec<Option<f64>> = keep.iter().map(|&i| family.lines[i].rho(&scaled(m))).collect();
                        (mean(&rho).unwrap_or(f64::NEG_INFINITY), m)
                    })
                    .collect();
                let best = scored.iter().map(|(r, _)| *r).fold(f64::NEG_INFINITY, f64::max);
                scored
                    .iter()
                    .filter(|(r, _)| *r >= best - 0.05)
                    .min_by(|a, b| a.1.ln().abs().total_cmp(&b.1.ln().abs()))
                    .map(|(_, m)| *m)
                    .unwrap_or(1.0)
            };
            let all: Vec<usize> = (0..family.lines.len()).collect();
            let folds: Vec<f64> = (0..family.lines.len())
                .map(|held| {
                    let keep: Vec<usize> = all.iter().copied().filter(|&i| i != held).collect();
                    best_over(&keep)
                })
                .collect();
            let held_out: Vec<Option<f64>> =
                folds.iter().enumerate().map(|(held, &m)| family.lines[held].rho(&scaled(m))).collect();
            Fitted {
                family: family.name,
                multipliers: folds,
                held_out: mean(&held_out),
                prior: family.rho(&device.planner),
                overall: best_over(&all),
            }
        })
        .collect()
}

/// Spearman's ρ of `a` against `b`: average ranks over ties, Pearson over
/// the ranks; `None` where either side holds no variance to correlate.
pub fn spearman(a: &[f64], b: &[f64]) -> Option<f64> {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (&x, &y) in ra.iter().zip(&rb) {
        cov += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    (va > 0.0 && vb > 0.0).then(|| cov / (va * vb).sqrt())
}

/// The ranks of `x`, ties averaged.
fn ranks(x: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
    let mut out = vec![0.0; x.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && x[order[j]] == x[order[i]] {
            j += 1;
        }
        for &at in &order[i..j] {
            out[at] = (i + j - 1) as f64 / 2.0;
        }
        i = j;
    }
    out
}

/// Replays every device's lines under `root` (a device `wanted` does not
/// name is left out, as is a family `ops` does not): the model's order of
/// each choice line the store holds against the ns it measured, offline,
/// no device opened. A shape's every salt that replays is correlated: two
/// layers of one shape (a plain and a biased epilogue, say) are two
/// measurements.
pub fn replay(root: &Path, wanted: Option<&str>, ops: Option<&[&str]>) -> Vec<Device> {
    let mut out = vec![];
    for (name, mut lines) in Lines::read(root) {
        if wanted.is_some_and(|w| !name.contains(w)) {
            continue;
        }
        let Some(target) = target(&name) else {
            eprintln!("{name}: no target of its arch; skipped");
            continue;
        };
        let Some(measured) = crate::probe::stored(root, &target) else {
            eprintln!("{name}: no readable store file; skipped");
            continue;
        };
        let unmeasured = measured.unmeasured.clone();
        let planner = std::sync::Arc::new(Planner::measured(target.clone(), measured).expect("an SM count"));
        let mut groups: Vec<String> = lines
            .entries
            .keys()
            .filter(|k| k.starts_with("choice|"))
            .map(|k| k.split('|').take(5).collect::<Vec<_>>().join("|"))
            .collect();
        groups.sort();
        groups.dedup();
        let mut kept: HashMap<&'static str, Family> = HashMap::new();
        for group in groups {
            let fields: Vec<&str> = group.split('|').collect();
            let (op, dtype, shape) = (fields[1], fields[3], fields[4]);
            let Some(dtype) = dtype_of(dtype) else { continue };
            if ops.is_some_and(|want| !want.contains(&family_of(op, dtype))) {
                continue;
            }
            let shape: Vec<usize> = shape.split('x').map(|n| n.parse().expect("a number")).collect();
            let Some(op) =
                ["linear", "gemm", "attention", "nearest", "conv2d"].into_iter().find(|known| *known == fields[1])
            else {
                continue;
            };
            let replays = match op {
                "linear" | "gemm" => gemm(&mut lines, &planner, op, dtype, &shape),
                "attention" => attention(&mut lines, &planner, dtype, &shape),
                "nearest" => nearest(&mut lines, &planner, dtype, &shape),
                _ => conv(&mut lines, &planner, dtype, &shape),
            };
            let name = family_of(op, dtype);
            kept.entry(name)
                .or_insert_with(|| Family::new(name))
                .lines
                .extend(replays.into_iter().filter(|r| r.timed.len() >= LEAST_CANDIDATES));
        }
        // What a family's lines came to: every choice line of its ops
        // correlated — derived, where the digest of a list this code
        // re-derives names it, else stored, its own candidates read under
        // the problem that admits them — too short to correlate, or
        // skipped: no hypothesis admits a candidate of it.
        let mut skipped_groups: Vec<(String, usize)> = vec![];
        let mut skipped: HashMap<String, usize> = HashMap::new();
        // The cross-replayed payloads already counted (builds write the same
        // measurement on several lines; the replay reads one).
        let mut counted: std::collections::HashSet<u64> = Default::default();
        for (key, entry) in &lines.entries {
            if !key.starts_with("choice|") {
                continue;
            }
            let fields: Vec<&str> = key.split('|').collect();
            let Some(dtype) = dtype_of(fields[3]) else { continue };
            if ops.is_some_and(|want| !want.contains(&family_of(fields[1], dtype))) {
                continue;
            }
            let Entry::Choice(timed) = entry else { continue };
            let family =
                kept.entry(family_of(fields[1], dtype)).or_insert_with(|| Family::new(family_of(fields[1], dtype)));
            family.total += 1;
            let payload = digest(timed);
            let cross = lines.stored.contains(&payload);
            if cross && !counted.insert(payload) {
                continue;
            }
            if lines.read.contains(key) || cross {
                if timed.len() >= LEAST_CANDIDATES {
                    family.replayed += 1;
                    if cross {
                        family.stored += 1;
                    }
                } else {
                    family.short += 1;
                }
            } else {
                let group = format!("{} {} {}", fields[1], fields[3], fields[4]);
                *skipped.entry(group).or_default() += 1;
            }
        }
        skipped_groups.extend(skipped);
        skipped_groups.sort();
        let mut families: Vec<Family> = kept.into_values().filter(|f| f.total > 0).collect();
        families.sort_by_key(|f| f.name);
        out.push(Device { name, unmeasured, planner, families, skipped_groups });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The family an op and dtype belong to; ops the harness does not replay
/// (norm, heads) count under `other`.
fn family_of(op: &str, dtype: ScalarDType) -> &'static str {
    match op {
        "linear" | "gemm" => "gemm",
        "attention" => "attention",
        "nearest" if dtype == ScalarDType::Int8 => "nearest i8",
        "nearest" => "nearest",
        "conv2d" => "conv",
        _ => "other",
    }
}

/// The dtype a line's key names.
fn dtype_of(name: &str) -> Option<ScalarDType> {
    match name {
        "Float16" => Some(ScalarDType::Float16),
        "BFloat16" => Some(ScalarDType::BFloat16),
        "Int8" => Some(ScalarDType::Int8),
        _ => None,
    }
}

/// The offline target of a device's name in the store ([`tune::device`]):
/// its arch and SM count, the dies of its arch's first listed card.
fn target(device: &str) -> Option<Target> {
    let (arch, sms) = device.rsplit_once('-')?;
    let sms = sms.strip_suffix("sm")?.parse().ok()?;
    let arch = match AmdArch::parse(arch) {
        Some(a) => GpuArch::Amd(a),
        None => GpuArch::Cuda(arch.trim_end_matches('a').parse::<CudaArch>().ok()?),
    };
    let xcds = CARDS.iter().find(|c| c.1 == arch).map_or(
        match arch {
            GpuArch::Amd(a) if a.is_cdna() => 8,
            _ => 1,
        },
        |c| c.3,
    );
    let target = Target::offline(arch, sms, xcds);
    (tune::device(&target) == device).then_some(target)
}

/// The stages of a shape's tuning over `ranked` ([`crate::ops::Tuner`]):
/// its admission kept, the candidates `expand` makes of them screened, and
/// the leading tiles' pipelines timed again; every stage's line keyed by
/// the digest of its list, a miss ending the attempt. One [`Replay`] per
/// choice line found.
fn stages<C: Tiled + std::fmt::Debug>(
    lines: &mut Lines,
    keying: &Keying,
    ranked: &[C],
    expand: impl Fn(&[C]) -> Vec<C>,
    configs: impl Fn(Vec<C>) -> Configs,
) -> Option<Vec<Replay>> {
    let reps = representatives(ranked);
    let at = lines.admitted(keying, "", &reps)?;
    let admitted: Vec<C> = at.iter().filter_map(|&i| reps.get(i).copied()).collect();
    let candidates = expand(&admitted);
    let screened = lines.choice(keying, " screened", &candidates)?;
    let mut out = vec![Replay { rule: " screened", configs: configs(candidates.clone()), timed: screened.clone() }];
    // The leading tiles' pipelines: as many as the contender band holds, at
    // most [`config::TWIN_TILES`] tiles, their twins in `ranked` judged and
    // the ones that compiled timed again.
    let ordered: Vec<(C, Option<u64>)> = screened.iter().map(|&(i, ns)| (candidates[i], Some(ns))).collect();
    let lead = ordered[0].1;
    let near: Vec<C> = ordered
        .iter()
        .filter(|(_, ns)| lead.zip(*ns).is_none_or(|(lead, ns)| ns as f64 <= lead as f64 * tune::CONTENDER))
        .map(|&(c, _)| c)
        .collect();
    let leaders: Vec<C> = config::distinct(&near).take(config::TWIN_TILES).collect();
    let twins: Vec<C> = config::distinct_configs(&leaders.iter().flat_map(|l| l.twins(ranked)).collect::<Vec<_>>());
    if twins.len() >= 2
        && let Some(judged) = lines.admitted(keying, " compiled", &twins)
    {
        let compiled: Vec<C> = judged.iter().filter_map(|&i| twins.get(i).copied()).collect();
        if compiled.len() > 1
            && let Some(whole) = lines.choice(keying, "", &compiled)
        {
            out.push(Replay { rule: "", configs: configs(compiled), timed: whole });
        }
    }
    Some(out)
}

/// The choice lines of `group` no derived list's digest matched, replayed
/// from their own candidates: each code the line carries beside its index
/// decodes to the config that ran, and the line is priced under the problem
/// of the hypothesis (of `hypotheses`, by index) whose admission holds the
/// most of them — a line no hypothesis admits any candidate of is left for
/// the skip count, its salt one the enumeration does not name. Its configs
/// list holds the candidates in the line's own timed order, its timed
/// indices that order's positions.
fn stored<C: Tiled>(
    lines: &mut Lines,
    group: &str,
    hypotheses: usize,
    admits: impl Fn(usize, &C) -> bool,
    configs: impl Fn(usize, Vec<C>) -> Configs,
) -> Vec<Replay> {
    let mut out = vec![];
    for (timed, payload) in lines.unread(group) {
        let Some(list) = timed.iter().map(|(_, c, _)| C::of_code(c)).collect::<Option<Vec<_>>>() else { continue };
        let held = |h: usize| list.iter().filter(|c| admits(h, c)).count();
        let Some((best, held)) = (0..hypotheses).map(|h| (h, held(h))).max_by_key(|&(_, n)| n) else { continue };
        if held == 0 || timed.len() < LEAST_CANDIDATES {
            continue;
        }
        lines.read.extend(
            lines
                .entries
                .iter()
                .filter(|(k, e)| of_group(k, group) && matches!(e, Entry::Choice(t) if digest(t) == payload))
                .map(|(k, _)| k.clone()),
        );
        lines.stored.insert(payload);
        let timed: Vec<(usize, u64)> = timed.iter().enumerate().map(|(at, &(_, _, ns))| (at, ns)).collect();
        out.push(Replay { rule: " stored", configs: configs(best, list), timed });
    }
    out
}

/// The GEMM family: `linear`'s lines, keyed by the batch and epilogue of
/// the layer that tuned them, and the device tests' and GEMM probes'
/// (`gemm`, [`DEVICE_TEST`]). Every salt that replays is a measurement.
fn gemm(lines: &mut Lines, planner: &Planner, op: &'static str, dtype: ScalarDType, shape: &[usize]) -> Vec<Replay> {
    let [cap, m, n, k] = shape[..] else { return vec![] };
    let batches = |bound: bool| if bound { cap.max(1) } else { 1 };
    let salts: Vec<(String, bool, bool)> = match op {
        "gemm" => [false, true]
            .into_iter()
            .flat_map(|gated| [false, true].map(move |bound| (format!("{DEVICE_TEST:?}"), gated, bound)))
            .collect(),
        _ => batches_of(cap)
            .into_iter()
            .flat_map(|batch| {
                let bound = matches!(batch, Batch::Var(_));
                epilogues().into_iter().map(move |epi| (format!("{:?}", (&batch, epi)), epi.gated, bound))
            })
            .collect(),
    };
    let mut out = vec![];
    let mut problems: Vec<GemmProblem> = vec![];
    for (salt, gated, bound) in salts {
        let keying = Keying::new(op, planner, dtype, shape, salt);
        let p = GemmProblem { batches: batches(bound), m, n, k, halves: if gated { 2 } else { 1 }, rope: None };
        let ranked = planner.gemm_ranked(p.batches, m, n, k, gated);
        let configs = |list: Vec<GemmCfg>| Configs::Gemm { problem: p, list };
        out.extend(stages(lines, &keying, &ranked, <[_]>::to_vec, configs).into_iter().flatten());
        if !problems.contains(&p) {
            problems.push(p);
        }
    }
    let (group, count) = (group(op, planner, dtype, shape), problems.len());
    out.extend(stored(
        lines,
        &group,
        count,
        |h, c| problems[h].admit(planner, c).is_ok(),
        |h, list| Configs::Gemm { problem: problems[h], list },
    ));
    out
}

/// A shape's batch as the op layer keys it: static at its capacity, or a
/// variable named `b` bound at one.
fn batches_of(cap: usize) -> Vec<Batch> {
    let mut out = vec![Batch::Static(cap)];
    if cap > 1 {
        out.push(Batch::var("b", 1, cap as i64));
    }
    out
}

/// The epilogues a tuned layer plausibly ran: the models' combinations of
/// bias, activation, gating and residual.
fn epilogues() -> Vec<Epilogue> {
    [Act::None, Act::Gelu, Act::Silu]
        .into_iter()
        .flat_map(|act| {
            [false, true].into_iter().flat_map(move |bias| {
                [false, true].into_iter().flat_map(move |gated| {
                    [false, true].map(move |residual| Epilogue {
                        bias,
                        act,
                        gated,
                        residual,
                        scale: None,
                        out_f32: false,
                        rope: None,
                    })
                })
            })
        })
        .collect()
}

/// The attention family: a shape's ranking admitted at its first clean
/// config, the softmax forms the tuner flipped (each pair's admission and
/// choice read back), and the tile stage over the formed ranking.
fn attention(lines: &mut Lines, planner: &Planner, dtype: ScalarDType, shape: &[usize]) -> Vec<Replay> {
    let [cap, t, tk, heads, _kv, d] = shape[..] else { return vec![] };
    let mut out = vec![];
    let mut problems: Vec<AttentionProblem> = vec![];
    for batch in batches_of(cap) {
        for causal in [false, true] {
            for bias in [None, Some(Bias::Shared), Some(Bias::PerBatch)] {
                let mask = AttnMask { causal, bias, ..AttnMask::default() };
                let salt = format!("{:?}", (&batch, mask, None::<usize>, None::<usize>, false, None::<usize>));
                let keying = Keying::new("attention", planner, dtype, shape, salt);
                let ranked = planner.attention_ranked(cap * heads, t, tk, d, causal, bias.is_some());
                let split = split_ranked(&ranked, None, tk);
                let reps = representatives(&split);
                let Some(at) = lines.first(&keying, &reps) else { continue };
                let Some(&start) = reps.get(at) else { return out };
                let mut forms = start.forms;
                for flip in crate::ops::attention::FORMS.iter().take(4 + usize::from(bias.is_some())) {
                    let pair = [FaCfg { forms, ..start }, FaCfg { forms: flip(forms), ..start }];
                    let Some(admitted) = lines.admitted(&keying, "", &pair) else { return out };
                    let list: Vec<FaCfg> = admitted.iter().filter_map(|&i| pair.get(i).copied()).collect();
                    let winner = match list.first() {
                        Some(&c) if list.len() == 1 => Some(c),
                        _ => lines.choice(&keying, "", &list).and_then(|c| c.first().map(|&(at, _)| list[at])),
                    };
                    let Some(winner) = winner else { return out };
                    forms = winner.forms;
                }
                let formed: Vec<FaCfg> = split.iter().map(|&c| FaCfg { forms, ..c }).collect();
                let p = AttentionProblem { tiles: cap * heads, t, tk, d, causal, bias: bias.is_some() };
                if !problems.contains(&p) {
                    problems.push(p);
                }
                let expand = |admitted: &[FaCfg]| {
                    admitted
                        .iter()
                        .flat_map(|c| {
                            planner
                                .splits(cap * heads * t.div_ceil(c.bq), tk.div_ceil(c.bkv))
                                .into_iter()
                                .map(move |splits| FaCfg { splits, ..*c })
                        })
                        .collect()
                };
                let configs = |list: Vec<FaCfg>| Configs::Attention { problem: p, list };
                out.extend(stages(lines, &keying, &formed, expand, configs).into_iter().flatten());
            }
        }
    }
    let (group, count) = (group("attention", planner, dtype, shape), problems.len());
    out.extend(stored(
        lines,
        &group,
        count,
        |h, c| problems[h].admit(planner, c).is_ok(),
        |h, list| Configs::Attention { problem: problems[h], list },
    ));
    out
}

/// The nearest family: the metric, floors and groups of the join that tuned
/// the shape are its salt, and where its rows or list ran live, their
/// bounds; the candidates the splits the planner adds to the admitted
/// configs.
fn nearest(lines: &mut Lines, planner: &Planner, dtype: ScalarDType, shape: &[usize]) -> Vec<Replay> {
    let (n, m, d, k) = match shape[..] {
        [n, m, d] => (n, m, d, 1),
        [n, m, d, k] => (n, m, d, k),
        _ => return vec![],
    };
    let var = |name: &str, max: usize| Var { name: name.into(), min: 1, max: max as i64 };
    let lives = [
        Live::default(),
        Live { rows: Some((var("n", n), 0)), k: None },
        Live { rows: None, k: (k > 1).then(|| var("k", k)) },
        Live { rows: Some((var("n", n), 0)), k: (k > 1).then(|| var("k", k)) },
    ];
    let mut out = vec![];
    let mut problems: Vec<NearestProblem> = vec![];
    for metric in [Metric::SquaredL2, Metric::ShiftedL2, Metric::L2, Metric::NegDot] {
        for floors in [
            Floors::default(),
            Floors { row: true, col: false },
            Floors { row: false, col: true },
            Floors { row: true, col: true },
        ] {
            for groups in [false, true] {
                for live in &lives {
                    let spec = NearestSpec {
                        n,
                        m,
                        d,
                        metric,
                        floors,
                        groups,
                        k,
                        cfg: NearestCfg::new(64, 64, 2),
                        live: live.clone(),
                    };
                    // The op layer's own salt, and the tuning probe's (a
                    // fresh store's cold tunes of these very shapes).
                    let salts =
                        [format!("{:?}", Salt(&spec)), format!("{:?}", (DEVICE_TEST, metric, floors, groups)).clone()];
                    let salts = if *live == Live::default() { salts.to_vec() } else { salts[..1].to_vec() };
                    let p = NearestProblem { n, m, d, dtype, metric, floors, groups, k };
                    if !problems.contains(&p) {
                        problems.push(p);
                    }
                    for salt in salts {
                        let keying = Keying::new("nearest", planner, dtype, shape, salt);
                        let ranked = planner.nearest_ranked(p);
                        let expand = |admitted: &[NearestCfg]| planner.nearest_splits(&p, admitted);
                        let configs = |list: Vec<NearestCfg>| Configs::Nearest { problem: p, list };
                        out.extend(stages(lines, &keying, &ranked, expand, configs).into_iter().flatten());
                    }
                }
            }
        }
    }
    let (group, count) = (group("nearest", planner, dtype, shape), problems.len());
    out.extend(stored(
        lines,
        &group,
        count,
        |h, c| problems[h].admit(planner, c).is_ok(),
        |h, list| Configs::Nearest { problem: problems[h], list },
    ));
    out
}

/// The convolution family: a line's shape is the full geometry the op keys
/// (the device tests' probes key its channel counts and side instead), the
/// candidates the splits a starved grid breaks into.
fn conv(lines: &mut Lines, planner: &Planner, dtype: ScalarDType, shape: &[usize]) -> Vec<Replay> {
    let geoms: Vec<(ConvGeom, usize, String)> = match shape[..] {
        [cin, cout, h, w, stride] => {
            let g =
                ConvGeom { h, w, cin, cout, kernel: [3, 3], stride: [stride, stride], pad: [1, 1], dilation: [1, 1] };
            vec![(g, 1, format!("{DEVICE_TEST:?}"))]
        }
        [images, h, w, cin, cout, k0, k1, s0, s1, p0, p1, d0, d1] => {
            let g = ConvGeom { h, w, cin, cout, kernel: [k0, k1], stride: [s0, s1], pad: [p0, p1], dilation: [d0, d1] };
            batches_of(images)
                .into_iter()
                .flat_map(|batch| epilogues().into_iter().map(move |epi| (g, images, format!("{:?}", (&batch, epi)))))
                .collect()
        }
        _ => return vec![],
    };
    let mut out = vec![];
    let mut problems: Vec<ConvProblem> = vec![];
    for (g, images, salt) in geoms {
        let [ho, wo] = g.out_hw();
        for bound in [false, true] {
            let salt = salt.clone();
            let (batches, m) = match bound {
                false => (1, images * ho * wo),
                true => (images, ho * wo),
            };
            let ranked = planner.conv_ranked(batches, m, &g);
            let p =
                ConvProblem { gemm: GemmProblem { batches, m, n: g.cout, k: g.k(), halves: 1, rope: None }, geom: g };
            if !problems.contains(&p) {
                problems.push(p);
            }
            let expand = |admitted: &[ConvCfg]| {
                planner.conv_splits(batches, m, &g, admitted).into_iter().filter(|c| c.split == 1 || !bound).collect()
            };
            let configs = |list: Vec<ConvCfg>| Configs::Conv { problem: p, list };
            let keying = Keying::new("conv2d", planner, dtype, shape, salt);
            out.extend(stages(lines, &keying, &ranked, expand, configs).into_iter().flatten());
        }
    }
    // The convolution's candidates are priced by their GEMM body's claim;
    // a hypothesis holds a candidate whose body it admits.
    let (group, count) = (group("conv2d", planner, dtype, shape), problems.len());
    out.extend(stored(
        lines,
        &group,
        count,
        |h: usize, c: &ConvCfg| problems[h].claim(planner, &c.gemm).is_ok(),
        |h, list| Configs::Conv { problem: problems[h], list },
    ));
    out
}

/// What a lowered program's instruction list counts by category: the
/// quantities the claim diff holds a config's program to. The list is the
/// code one warp runs, its loops rolled where a body holds a product, so a
/// rolled trip's counts sit in it once; the fill counts every thread's
/// loads (`cp.async` copies by their width, on the targets that fill
/// through them).
pub struct Counted {
    /// The whole program's instruction count.
    pub instructions: usize,
    /// The shared bytes the program declares.
    pub smem: usize,
    /// Matrix instructions (`wmma`).
    pub steps: usize,
    /// VALU operations (arithmetic, logic, casts).
    pub valu: usize,
    /// The bytes of global memory the program's loads move.
    pub fill: usize,
}

/// What `program`'s linear list holds, by category.
pub fn counted(program: &svod_ir::UOp) -> Option<Counted> {
    use svod_dtype::AddrSpace;
    use svod_ir::Op;
    use svod_ir::ops;
    let Op::Program(ops::Program { linear: Some(linear), .. }) = program.op() else { return None };
    let list: Vec<_> = linear.op().sources().to_vec();
    // A buffer's address space, and the shared bytes its declaration holds.
    let mut spaces: HashMap<u64, AddrSpace> = HashMap::new();
    let mut smem = 0;
    for u in &list {
        let arg = match u.op() {
            Op::Buffer(ops::Buffer { arg, .. }) | Op::Param(ops::Param { arg, .. }) => arg,
            _ => continue,
        };
        let Some(space) = arg.addrspace else { continue };
        if space == AddrSpace::Local {
            match u.op().sources()[0].op() {
                Op::Const(elems) => {
                    smem += match elems.0 {
                        svod_ir::ConstValue::Int(elems) => elems.max(0) as usize * u.dtype().bytes(),
                        _ => 0,
                    }
                }
                _ => return None,
            }
        }
        spaces.insert(u.id, space);
    }
    // The buffer an access reads (a view chain), for the space it loads
    // from, and the elements it spans: an INDEX one, a SHRINK its size.
    let space_of = |u: &std::sync::Arc<svod_ir::UOp>| -> Option<AddrSpace> {
        let mut at = u;
        for _ in 0..8 {
            match at.op() {
                Op::Buffer(..) | Op::Param(..) => return spaces.get(&at.id).copied(),
                Op::Shrink(ops::Shrink { src, .. }) => at = src,
                Op::Slice(ops::Slice { buffer, .. }) | Op::Index(ops::Index { buffer, .. }) => at = buffer,
                Op::GetAddr(ops::GetAddr { src, .. }) => at = src,
                _ => return None,
            }
        }
        None
    };
    let width_of = |index: &std::sync::Arc<svod_ir::UOp>| match index.op() {
        Op::Shrink(ops::Shrink { sizes, .. }) => match sizes.op() {
            Op::Const(elems) => match elems.0 {
                svod_ir::ConstValue::Int(w) => w.max(1) as usize,
                _ => 1,
            },
            _ => 1,
        },
        _ => 1,
    };
    let mut out = Counted { instructions: list.len(), smem, steps: 0, valu: 0, fill: 0 };
    // The loops the statement at hand sits in: the bytes a trip moves are
    // what a loop body loads — a block's once-over reads (its resident rows)
    // sit outside them, priced as the working set they join, not a trip's.
    let mut loops = 0usize;
    for u in &list {
        match u.op() {
            Op::Wmma(..) => out.steps += 1,
            Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) | Op::Cast(..) | Op::BitCast(..) => out.valu += 1,
            Op::Range(..) => loops += 1,
            Op::End(ops::End { ranges, .. }) => loops = loops.saturating_sub(ranges.len()),
            // A `cp.async` copy moves its width's bytes from global memory
            // without a load the list would hold; the commit and wait groups
            // a ring lands them with move nothing.
            Op::Custom(ops::Custom { code, .. }) if code.contains("cp.async") && code.contains("shared.global") => {
                out.fill += 16;
            }
            // A plain load inside a loop body moves its access's width of
            // elements — the fill's own on a staged target, a side read's
            // beside it.
            Op::Load(ops::Load { index, .. }) if loops > 0 && space_of(index) == Some(AddrSpace::Global) => {
                out.fill += u.dtype().bytes() * width_of(index);
            }
            _ => {}
        }
    }
    Some(out)
}

/// A config's predicted resource claim — the quantities the cost model
/// counts, read from the family's own [`config::Claim`] construction (the
/// one path the ranking prices), outside the fused ns: what a trip of the
/// config's kernel should hold. The VALU count keeps the epilogue's own
/// units (a pair's operations, a listing's share), which a program's
/// instruction mix only sometimes maps.
pub struct Claim {
    /// Matrix-core steps a trip, in the target's atom instructions.
    pub steps: f64,
    /// VALU operations a trip.
    pub ops: f64,
    /// The bytes a trip reads from global memory: its ring's fill and its
    /// side reads alike, the loads the program's list counts by address
    /// space.
    pub fill: usize,
    /// The shared memory the config declares.
    pub smem: usize,
    /// The registers a lane holds by the layout count.
    pub regs: u32,
}

/// One config's claim against its program: per quantity the predicted
/// value, the program's own count, and whether they hold within ±15 % —
/// a quantity that cannot map is reported, not forced.
pub struct Diffed {
    pub name: String,
    /// `(quantity, predicted, program, holds)`.
    pub rows: Vec<(&'static str, f64, f64, bool)>,
    /// The registers a lane holds by the layout count: the linear list
    /// holds no register count (the compiler's verdict, not the program's).
    pub regs: u32,
}

/// How far a program's count may sit from the claim it checks.
const HOLDS: f64 = 0.15;

/// A sample of each family's configs emitted and counted against the
/// model's predictions — the GEMM tiles of a 4096³ plan, two attention
/// configs, nearest at `k = 1` resident and streamed, a register list at
/// `k = 64` and a split at `k = 8` — on `target`'s planner under the
/// declared guessed costs. The tensors are small (the interpreter's
/// shapes); the tiles are those a large shape plans. Every row is the
/// family's own claim, the same construction the ranking prices. The VALU
/// row's units stay each family's: a nearest or attention pair's bundle in
/// score elements (the claim's counts folded at the epilogue's unit), a
/// GEMM's epilogue un-amortized over its trips, its tile operations an
/// instruction a lane.
pub fn claims(target: &crate::atoms::Target) -> (Vec<Diffed>, Vec<Diffed>) {
    use crate::build::{BF16, F16};
    use crate::kernels::attention::{AttnMask, AttnSpec, attention};
    use crate::kernels::gemm::{Epilogue, GemmSpec, gemm};
    use crate::kernels::nearest::{Live, Metric, NearestSpec, nearest};
    use crate::kernels::{Act, Batch};

    let device = crate::atoms::Device { target: target.clone(), costs: crate::probe::GUESSED };
    let planner = Planner::new(device.clone()).expect("an SM count");
    let lowering = crate::lower::Lowering::new(&device);
    let atom = target.geometry().expect("a matrix core");
    let atom_step = (atom.m as usize * atom.n as usize * atom.k as usize) as f64;
    // The program's list is one warp's code, its loads one thread's: the
    // block's per-trip claim divides by the warps for the steps and the
    // VALU, and its fill bytes compare against the threads' loads together.
    let diff =
        |name: String, claim: &config::Claim, valu: f64, warps: usize, prog: crate::ir::Program| -> Option<Diffed> {
            let counted = counted(&crate::lower::standalone(prog, &lowering).ok()?.program)?;
            let threads = (warps * target.wave as usize) as f64;
            let predicted = Claim {
                steps: claim.steps.count() * config::STEP as f64 / atom_step,
                ops: valu,
                fill: (claim.fill.bytes + claim.side.bytes) as usize,
                smem: claim.smem,
                regs: claim.regs,
            };
            let Claim { steps, ops, fill, smem, regs } = predicted;
            let row = |what: &'static str, predicted: f64, program: f64| {
                (what, predicted, program, (program - predicted).abs() <= HOLDS * predicted.max(1.0))
            };
            Some(Diffed {
                name,
                regs,
                rows: vec![
                    row("steps/trip", steps / warps as f64, counted.steps as f64),
                    row("fill bytes/trip", fill as f64, counted.fill as f64 * threads),
                    row("valu ops/trip", ops, counted.valu as f64 * warps as f64),
                    row("smem", smem as f64, counted.smem as f64),
                ],
            })
        };
    let mut out: Vec<(String, config::Claim, f64, usize, crate::ir::Program)> = vec![];
    // The static decomposition rows (see the k64 list below): what the
    // program's listing adder holds against the counted parts.
    let mut statics: Vec<Diffed> = vec![];
    // GEMM: the leading tiles of a 4096³ plan, emitted small, a gelu
    // epilogue on the accumulator — the claim's own epilogue term, un-
    // amortized over the trips and in the static-code units a program's
    // instruction mix counts (the lane operations over a wave).
    let p = GemmProblem { batches: 1, m: 4096, n: 4096, k: 4096, halves: 1, rope: None };
    for cfg in config::distinct(&planner.gemm_ranked(1, 4096, 4096, 4096, false)).take(3) {
        let claim = p.claim(&planner, &cfg).expect("ranked");
        let valu = claim.ops.epilogue * (4096 / cfg.tile[2]) as f64 / target.wave as f64;
        let spec = GemmSpec {
            m: 130,
            n: 192,
            k: cfg.tile[2] * 8,
            batch: Batch::Static(1),
            epilogue: Epilogue { act: Act::Gelu, ..Epilogue::DEFAULT },
            cfg,
        };
        out.push((
            format!("gemm {:?}", cfg.tile),
            claim,
            valu,
            (cfg.warps[0] * cfg.warps[1]) as usize,
            gemm::<BF16>(&spec),
        ));
    }
    // Attention: two configs of a d64 causal plan, emitted small.
    let p = AttentionProblem { tiles: 8, t: 100, tk: 100, d: 64, causal: true, bias: false };
    for cfg in config::distinct(&planner.attention_ranked(8, 100, 100, 64, true, false)).take(2) {
        let claim = p.claim(&planner, &cfg).expect("ranked");
        let valu = claim.ops.fold(config::pair_op(&device), 1.0, &config::Weights::same());
        let spec = AttnSpec {
            batch: Batch::Static(2),
            t: 100,
            tk: 100,
            heads: 4,
            kv_heads: 2,
            d: 64,
            mask: AttnMask { causal: true, ..AttnMask::default() },
            cache: None,
            packed: false,
            scale: 0.125,
            cfg,
        };
        out.push((
            format!("attention {}x{}", cfg.bq, cfg.bkv),
            claim,
            valu,
            cfg.warps() as usize,
            attention::<BF16>(&spec),
        ));
    }
    // Nearest: k = 1 resident and streamed, a register list at k = 64, a
    // split at k = 8.
    let problem = |k| NearestProblem {
        n: 1 << 16,
        m: 1 << 16,
        d: 128,
        dtype: ScalarDType::Float16,
        metric: Metric::SquaredL2,
        floors: Floors::default(),
        groups: false,
        k,
    };
    let spec = |cfg, k| NearestSpec {
        n: 100,
        m: 960,
        d: 128,
        metric: Metric::SquaredL2,
        floors: Floors::default(),
        groups: false,
        k,
        cfg,
        live: Live::default(),
    };
    let ranked = planner.nearest_ranked(problem(1));
    for (name, way) in [
        ("resident", ranked.iter().find(|c| c.bk.is_none()).copied()),
        ("streamed", ranked.iter().find(|c| c.bk.is_some()).copied()),
    ] {
        let Some(cfg) = way else { continue };
        let claim = problem(1).claim(&planner, &cfg).expect("ranked");
        let valu = claim.ops.fold(config::pair_op(&device), 1.0, &config::Weights::same());
        out.push((
            format!("nearest k1 {name} {}x{}", cfg.bq, cfg.bkv),
            claim,
            valu,
            cfg.warps() as usize,
            nearest::<F16>(&spec(cfg, 1)),
        ));
    }
    let ranked = planner.nearest_ranked(problem(64));
    if let Some(&cfg) = ranked.iter().find(|c| c.queue.is_none() && c.select.is_none()) {
        let claim = problem(64).claim(&planner, &cfg).expect("ranked");
        let valu = claim.ops.fold(config::pair_op(&device), 1.0, &config::Weights::same());
        // The listing's static decomposition beside the expected-ops row:
        // the program's listing adder (its count less the same tile's at
        // `k = 1`, where no list runs) against the parts
        // [`NearestProblem::listing_static`] counts, within ±30 % — the
        // guarded bodies, the merge network's stages, the queues' sort.
        let twin = crate::kernels::nearest::NearestCfg { lanes: None, queue: None, select: None, ..cfg };
        let listing = |prog: crate::ir::Program| {
            counted(&crate::lower::standalone(prog, &lowering).ok()?.program).map(|c| c.valu as f64)
        };
        if let Some(program) = listing(nearest::<F16>(&spec(cfg, 64)))
            .zip(listing(nearest::<F16>(&spec(twin, 1))))
            .map(|(whole, least)| whole - least)
        {
            let predicted = problem(64).listing_static(&cfg);
            statics.push(Diffed {
                name: format!("nearest k64 list {}x{}", cfg.bq, cfg.bkv),
                regs: 0,
                rows: vec![(
                    "listing static",
                    predicted,
                    program,
                    (program - predicted).abs() <= HOLDS_STATIC * predicted.max(1.0),
                )],
            });
        }
        out.push((
            format!("nearest k64 list {}x{}", cfg.bq, cfg.bkv),
            claim,
            valu,
            cfg.warps() as usize,
            nearest::<F16>(&spec(cfg, 64)),
        ));
    }
    let split_problem = NearestProblem { n: 256, m: 1 << 20, ..problem(8) };
    let ranked = planner.nearest_ranked(split_problem);
    let split = planner.nearest_splits(&split_problem, &config::distinct(&ranked).take(16).collect::<Vec<_>>());
    if let Some(&cfg) = split.iter().find(|c| c.splits > 1 && c.splits <= 32) {
        let claim = split_problem.claim(&planner, &cfg).expect("ranked");
        let valu = claim.ops.fold(config::pair_op(&device), 1.0, &config::Weights::same());
        let mut split_spec = spec(cfg, 8);
        split_spec.m = cfg.splits * cfg.bkv * 2;
        out.push((
            format!("nearest k8 split {} {}x{}", cfg.splits, cfg.bq, cfg.bkv),
            claim,
            valu,
            cfg.warps() as usize,
            nearest::<F16>(&split_spec),
        ));
    }
    let diffed = out
        .into_iter()
        .filter_map(|(name, claim, gelu_ops, warps, prog)| diff(name, &claim, gelu_ops, warps, prog))
        .collect();
    (diffed, statics)
}

/// How far a program's static listing adder may sit from the parts the
/// decomposition counts: the grid the constants were measured on holds
/// within 27 %, the diff's own sample within 7 %.
const HOLDS_STATIC: f64 = 0.30;
