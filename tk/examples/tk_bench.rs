//! Device benchmarks of tk's kernels on the default device (`SVOD_DEVICE`),
//! every time the fastest stamped run of the device's own clock (its kernels
//! summed) after the clock has lifted, plans timed round-robin so drift hits
//! each alike ([`best_of`]).
//!
//! ```text
//! cargo run -p svod-tk --release --example tk_bench -- <command> [options]
//! ```
//!
//! - `gemm [--shape MxNxK,…] [--top N]`: bf16 GEMM TFLOP/s of the tuner's
//!   pick (4096³ by default) and the planner's first `N` configs.
//! - `attention [--shape BxHxTxD,…] [--causal] [--bias] [--top N]`: flash
//!   attention TFLOP/s of what the op launches, by default B 4, H 8, T 2048
//!   at d64 (plain, causal, a per-batch bias, both) and d128 (plain, causal).
//! - `nearest [--rows N] [--depths D,…] [--k K] [--metric M]`: a self-join's
//!   pairs a second and TFLOP/s through the op, by default 65536 rows at
//!   depths 16–128: squared distances, the same under 64 groups, L2 above
//!   both floors (`--metric` alone instead).
//! - `rerank [--rows N] [--candidates M] [--depths D,…] [--ks K,…] [--metric M]`:
//!   `k_nearest_reranked` over its 8-bit and its f16 search against the
//!   plain f16 op of `k`.
//! - `raster [--shape MxNxK,…]`: every raster of the tuner's GEMM pick and
//!   of the planner's first config against the default's, beside the
//!   model's price of each.
//! - `fit [--only LABEL,…] [--picks]`: the cost model against the device:
//!   one `fit|label|rank|verdict|ns predicted|ns measured|cfg` line a ranked
//!   config of a few GEMM, attention and nearest shapes and a
//!   `pick|label|ns|cfg` line a shape, the offline fit's rank correlation
//!   and regret, and `tune|label|s` the tuner's wall time on the store (cold
//!   on a store of the device's probes alone: the spread `PRUNED_PAST` is
//!   chosen from); `--picks` times the picks alone, an A/B across builds.
//! - `trips`: the trip probes at 1–48 blocks an SM, per block-trip and per
//!   round ([`svod_tk::probe::trip_at`]): what `TRIP_HIDING` is fitted to.
//! - `pin --cfg BMxBNxBK:S:WRxWC[:rolled],… [--shape MxNxK,…] [--residual]
//!   [--gated]`: each pinned GEMM config on each shape (Whisper's fc2 by
//!   default, a SwiGLU epilogue under `--gated`),
//!   timed beside the model's price and that price with one term zeroed at
//!   a time (the memory latency, the lone trip, the matrix step): what each
//!   term holds of the config's time.
//!
//! `--rounds R` sets the timed rounds (default 20; `fit` 2).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use svod_dtype::default_device::default_device;
use svod_dtype::{DType, ScalarDType};
use svod_runtime::ExecutionPlan;
use svod_runtime::benchmark::{CLOCK_WARMUP, round_robin_min, warm_clock};
use svod_tensor::Tensor;
use svod_tk::atoms::{Core, Costs, Device};
use svod_tk::build::{BF16, F16};
use svod_tk::ir::{ParamKind, Program, Schedule};
use svod_tk::kernels::attention::{AttnMask, AttnSpec, Bias, attention};
use svod_tk::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, TileOrder, gemm};
use svod_tk::kernels::nearest::{Floors, Metric, NearestSpec, nearest};
use svod_tk::kernels::{Act, Batch};
use svod_tk::lower::Lowering;
use svod_tk::ops::config::{AttentionProblem, GemmProblem, NearestProblem, Planner};
use svod_tk::ops::shape::{self, Extent, Plan};
use svod_tk::ops::{self, Attn, Nearest, Quantized, Rerank, Search, tuned};
use svod_tk::tune::TuneStore;

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

const USAGE: &str = "tk_bench <gemm|attention|nearest|rerank|raster|fit|trips|pin> [options]; see the example's docs";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first().filter(|a| !a.starts_with('-')) else {
        println!("{USAGE}");
        return Ok(());
    };
    // The tuner's picks and round times under `RUST_LOG=svod_tk=debug`.
    let filter = tracing_subscriber::EnvFilter::from_default_env();
    tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).without_time().init();
    let args = Args(&args[1..]);
    match command.as_str() {
        "gemm" => gemm_bench(&args),
        "attention" => attention_bench(&args),
        "nearest" => nearest_bench(&args),
        "rerank" => rerank_bench(&args),
        "raster" => raster_bench(&args),
        "fit" => fit_bench(&args),
        "trips" => trips_bench(),
        "pin" => pin_bench(&args),
        other => Err(format!("{other}: {USAGE}").into()),
    }
}

struct Args<'a>(&'a [String]);

impl Args<'_> {
    fn opt(&self, name: &str) -> Option<&str> {
        let at = self.0.iter().position(|a| a == name)?;
        self.0.get(at + 1).map(String::as_str)
    }

    fn flag(&self, name: &str) -> bool {
        self.0.iter().any(|a| a == name)
    }

    fn num(&self, name: &str, default: usize) -> Result<usize> {
        Ok(self.opt(name).map(str::parse).transpose()?.unwrap_or(default))
    }

    /// A comma-separated list of counts.
    fn counts(&self, name: &str, default: &str) -> Result<Vec<usize>> {
        Ok(self.opt(name).unwrap_or(default).split(',').map(|v| v.trim().parse()).collect::<Result<_, _>>()?)
    }

    /// A comma-separated list of shapes, their dims split by `x`.
    fn shapes<const N: usize>(&self, default: &str) -> Result<Vec<[usize; N]>> {
        let shape = |s: &str| -> Result<[usize; N]> {
            let dims: Vec<usize> = s.split('x').map(|d| d.trim().parse()).collect::<Result<_, _>>()?;
            dims.try_into().map_err(|d| format!("{d:?}: {N} dims").into())
        };
        self.opt("--shape").unwrap_or(default).split(',').map(shape).collect()
    }

    fn metric(&self) -> Result<Option<Metric>> {
        Ok(match self.opt("--metric") {
            None => None,
            Some("squared-l2") => Some(Metric::SquaredL2),
            Some("l2") => Some(Metric::L2),
            Some("neg-dot") => Some(Metric::NegDot),
            Some(other) => return Err(format!("--metric {other}: squared-l2, l2 or neg-dot").into()),
        })
    }

    fn rounds(&self, default: usize) -> Result<usize> {
        self.num("--rounds", default)
    }
}

/// The op layer's planner of the default device: its cost probes measured
/// (or read from the tune store) once.
fn planner() -> Result<Arc<Planner>> {
    ops::planner(&default_device()).ok_or_else(|| "no GPU tk has kernels for: set SVOD_DEVICE".into())
}

/// One run's device time, its kernels' stamps summed; `None` when it failed
/// or a kernel lost a stamp.
fn stamped(plan: &ExecutionPlan) -> Option<Duration> {
    let kernels = plan.execute_profiled().ok()?;
    let ns: Option<u64> = kernels.iter().map(|k| k.gpu_end_ns?.checked_sub(k.gpu_start_ns?)).sum();
    (!kernels.is_empty()).then_some(Duration::from_nanos(ns?))
}

/// Each plan's fastest [`stamped`] run over `rounds` round-robin rounds,
/// after the first lifted the clock.
fn best_of(plans: &[ExecutionPlan], rounds: usize) -> Vec<Option<Duration>> {
    if let Some(first) = plans.first() {
        warm_clock(CLOCK_WARMUP, || stamped(first));
    }
    round_robin_min(plans.len(), rounds, None, |i| stamped(&plans[i]))
}

/// Realized tensors of drawn values for the programs a bench launches,
/// shared by every program naming a parameter alike, outputs included.
#[derive(Default)]
struct Buffers(HashMap<(String, ScalarDType, usize), Tensor>);

impl Buffers {
    fn of(&mut self, prog: &Program) -> Result<Vec<Tensor>> {
        prog.params()
            .iter()
            .map(|p| {
                let key = (p.name.clone(), p.dtype, p.elems);
                if let Some(t) = self.0.get(&key) {
                    return Ok(t.clone());
                }
                let t = match p.kind {
                    ParamKind::In => drawn(&[p.elems], DType::Scalar(p.dtype))?,
                    _ => Tensor::empty(&[p.elems], DType::Scalar(p.dtype)),
                };
                Ok(self.0.entry(key).or_insert(t).clone())
            })
            .collect()
    }

    /// `prog` launched under `lowering` on these buffers, prepared.
    fn plan(&mut self, prog: Program, lowering: &Lowering) -> Result<ExecutionPlan> {
        let tensors = self.of(&prog)?;
        let refs: Vec<&Tensor> = tensors.iter().collect();
        Ok(svod_tk::launch::graph_launch(prog, lowering, &refs)?.prepare()?)
    }
}

/// A realized tensor of uniform values in `dtype`.
fn drawn(shape: &[usize], dtype: DType) -> Result<Tensor> {
    let t = Tensor::rand_with(shape, DType::Float32, default_device())?.cast(dtype).contiguous();
    t.realize()?;
    Ok(t)
}

/// `ns` as ms and the rate of `work` a ns (TFLOP/s for flops, Gpairs/s for pairs).
fn rate(t: Option<Duration>, work: f64, unit: &str) -> String {
    t.map_or("failed".into(), |t| {
        let ns = t.as_nanos() as f64;
        format!("{:9.3} ms {:7.2} {unit}", ns / 1e6, work / ns / 1e3)
    })
}

fn gemm_label(c: &GemmCfg) -> String {
    format!("{:?} s{} {:?} unroll={} {:?} crews={}", c.tile, c.stages(), c.warps, c.unroll(), c.core, c.crews())
}

/// A plain bf16 GEMM at `cfg`, admitted.
fn gemm_program(planner: &Planner, [m, n, k]: [usize; 3], epilogue: Epilogue, cfg: GemmCfg) -> Result<Program> {
    let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue };
    let spec = spec.admitted(planner, cfg).map_err(|why| format!("{cfg:?}: {why:?}"))?;
    Ok(gemm::<BF16>(&spec))
}

/// The tuner's GEMM pick for `shape`, as the device tests and GEMM probes
/// tune ([`tuned::gemm_probe`]).
fn gemm_pick(planner: &Planner, [m, n, k]: [usize; 3], epilogue: Epilogue, ranked: &[GemmCfg]) -> GemmCfg {
    let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue };
    tuned::gemm_probe(TuneStore::global(), planner, DType::BFloat16, ranked, spec).cfg()
}

fn gemm_bench(args: &Args) -> Result<()> {
    let (top, rounds) = (args.num("--top", 0)?, args.rounds(20)?);
    let planner = planner()?;
    for [m, n, k] in args.shapes::<3>("4096x4096x4096")? {
        let ranked = planner.gemm_ranked(1, m, n, k, &Epilogue::DEFAULT);
        let pick = gemm_pick(&planner, [m, n, k], Epilogue::DEFAULT, &ranked);
        let mut cfgs: Vec<GemmCfg> = ranked.iter().take(top).copied().collect();
        if !cfgs.contains(&pick) {
            cfgs.push(pick);
        }
        let mut buffers = Buffers::default();
        let plans = cfgs
            .iter()
            .map(|&c| {
                buffers.plan(gemm_program(&planner, [m, n, k], Epilogue::DEFAULT, c)?, &c.lowering(&planner.device))
            })
            .collect::<Result<Vec<_>>>()?;
        println!("== gemm {m}x{n}x{k} bf16");
        for (c, t) in cfgs.iter().zip(best_of(&plans, rounds)) {
            let mark = if *c == pick { " (tuner's pick)" } else { "" };
            println!("  {}: {}{mark}", gemm_label(c), rate(t, 2.0 * (m * n * k) as f64, "TFLOP/s"));
        }
    }
    Ok(())
}

fn attention_bench(args: &Args) -> Result<()> {
    let (top, rounds) = (args.num("--top", 0)?, args.rounds(20)?);
    let cases: Vec<([usize; 4], bool, bool)> = match args.opt("--shape") {
        Some(_) => args.shapes::<4>("")?.into_iter().map(|s| (s, args.flag("--causal"), args.flag("--bias"))).collect(),
        None => [
            (64, false, false),
            (64, false, true),
            (64, true, false),
            (64, true, true),
            (128, false, false),
            (128, true, false),
        ]
        .map(|(d, causal, bias)| ([4, 8, 2048, d], causal, bias))
        .to_vec(),
    };
    let planner = planner()?;
    for ([b, h, t, d], causal, bias) in cases {
        let (q, k, v) = (
            drawn(&[b, t, h, d], DType::BFloat16)?,
            drawn(&[b, t, h, d], DType::BFloat16)?,
            drawn(&[b, t, h, d], DType::BFloat16)?,
        );
        let scores = if bias { Some(drawn(&[b, h, t, t], DType::BFloat16)?) } else { None };
        let o = ops::attention(&q, &k, &v, Attn { causal, bias: scores.as_ref(), ..Attn::default() })?;
        let launched = match ops::chosen() {
            Some(ops::Chosen::Kernel { cfg, .. }) => cfg,
            other => format!("{other:?}"),
        };
        let mut labels = vec![format!("the op: {launched}")];
        let mut plans = vec![o.prepare()?];
        let mask = AttnMask { causal, bias: bias.then_some(Bias::PerBatch), ..AttnMask::default() };
        let spec = AttnSpec {
            batch: Batch::Static(b),
            t,
            tk: t,
            heads: h,
            kv_heads: h,
            d,
            mask,
            cache: None,
            packed: false,
            scale: 1.0 / (d as f32).sqrt(),
        };
        let mut buffers = Buffers::default();
        for &c in planner.attention_ranked(b * h, t, t, d, causal, bias).iter().take(top) {
            let admitted = spec.clone().admitted(&planner, c).map_err(|why| format!("{c:?}: {why:?}"))?;
            plans.push(buffers.plan(attention::<BF16>(&admitted), &c.lowering(&planner.device))?);
            labels.push(format!("ranked {c:?}"));
        }
        let flops = 4.0 * (b * h * t * t * d) as f64 / if causal { 2.0 } else { 1.0 };
        println!("== attention b{b} h{h} t{t} d{d} causal={causal} bias={bias}");
        for (label, time) in labels.iter().zip(best_of(&plans, rounds)) {
            println!("  {}: {label}", rate(time, flops, "TFLOP/s"));
        }
    }
    Ok(())
}

fn nearest_bench(args: &Args) -> Result<()> {
    let (n, k, rounds) = (args.num("--rows", 1 << 16)?, args.num("--k", 1)?, args.rounds(20)?);
    let depths = args.counts("--depths", "16,32,64,128")?;
    let both = Floors { row: true, col: true };
    let none = Floors { row: false, col: false };
    let variants = match args.metric()? {
        Some(metric) => vec![(metric, none, false)],
        None => vec![(Metric::SquaredL2, none, false), (Metric::SquaredL2, none, true), (Metric::L2, both, false)],
    };
    let floor = drawn(&[n], DType::Float32)?;
    let group = Tensor::from_slice((0..n as i32).map(|i| i % 64).collect::<Vec<_>>());
    group.realize()?;
    for d in depths {
        let x = drawn(&[n, d], DType::BFloat16)?;
        for &(metric, floors, groups) in &variants {
            let opts = Nearest {
                metric,
                row_floor: floors.row.then_some(&floor),
                col_floor: floors.col.then_some(&floor),
                groups: groups.then_some((&group, &group)),
            };
            let (_, index) = if k == 1 { ops::nearest(&x, &x, opts)? } else { ops::k_nearest(&x, &x, k, opts)? };
            let chosen = ops::chosen();
            let [time] = best_of(&[index.prepare()?], rounds)[..] else { unreachable!("one plan") };
            let pairs = (n * n) as f64;
            let flops = time
                .map_or("".into(), |t| format!(" {:6.2} TFLOP/s", 2.0 * d as f64 * pairs / t.as_nanos() as f64 / 1e3));
            println!(
                "nearest {n}x{n}x{d} k{k} {metric:?} {floors:?} groups={groups}: {}{flops} {chosen:?}",
                rate(time, pairs, "Gpairs/s")
            );
        }
    }
    Ok(())
}

fn rerank_bench(args: &Args) -> Result<()> {
    let (n, rounds) = (args.num("--rows", 16384)?, args.rounds(20)?);
    let m = args.num("--candidates", n)?;
    let (depths, ks) = (args.counts("--depths", "128,256,512,768")?, args.counts("--ks", "1,16,64")?);
    let metric = args.metric()?.unwrap_or(Metric::NegDot);
    let opts = Nearest { metric, ..Nearest::default() };
    for d in depths {
        let (x, c) = (drawn(&[n, d], DType::Float32)?, drawn(&[m, d], DType::Float32)?);
        let (x16, c16) = (x.cast(DType::Float16).contiguous(), c.cast(DType::Float16).contiguous());
        let (x8, c8) = (Quantized::new(&x)?, Quantized::new(&c)?);
        for t in [&x16, &c16, &x8.rows, &x8.scale, &c8.rows, &c8.scale] {
            t.realize()?;
        }
        for &k in &ks {
            let mut plans = vec![];
            let mut picks = vec![];
            let (_, index) = ops::k_nearest(&x16, &c16, k, opts)?;
            plans.push(index.prepare()?);
            picks.push(ops::chosen());
            for search in [Search::Int8, Search::Half(ScalarDType::Float16)] {
                let rerank = match search {
                    Search::Int8 => Rerank { x8: Some(&x8), c8: Some(&c8), search, ..Rerank::default() },
                    _ => Rerank { search, ..Rerank::default() },
                };
                let (_, index) = ops::k_nearest_reranked(&x, &c, k, opts, rerank)?;
                plans.push(index.prepare()?);
                picks.push(ops::chosen());
            }
            let times = best_of(&plans, rounds);
            let pairs = (n * m) as f64;
            let ratio =
                |t: Option<Duration>| times[0].zip(t).map_or(f64::NAN, |(f, t)| f.as_secs_f64() / t.as_secs_f64());
            println!(
                "rerank {n}x{m} d{d} {metric:?} k{k}: f16 {} | int8 reranked {} ({:.2}x f16) | f16 reranked {} ({:.2}x f16) | {picks:?}",
                rate(times[0], pairs, "Gpairs/s"),
                rate(times[1], pairs, "Gpairs/s"),
                ratio(times[1]),
                rate(times[2], pairs, "Gpairs/s"),
                ratio(times[2]),
            );
        }
    }
    Ok(())
}

fn raster_bench(args: &Args) -> Result<()> {
    let rounds = args.rounds(20)?;
    let planner = planner()?;
    for [m, n, k] in args.shapes::<3>("4096x4096x4096,4096x1024x4096,8192x8192x1024")? {
        let ranked = planner.gemm_ranked(1, m, n, k, &Epilogue::DEFAULT);
        let pick = gemm_pick(&planner, [m, n, k], Epilogue::DEFAULT, &ranked);
        let first = ranked[0];
        let p = GemmProblem::new(1, m, n, k, &Epilogue::DEFAULT);
        let flops = 2.0 * (m * n * k) as f64;
        for (what, cfg) in [("tuner's pick", pick), ("planner's first", first)] {
            if what == "planner's first" && (GemmCfg { order: pick.order, ..first }) == pick {
                continue;
            }
            let gm = m.div_ceil(cfg.tile[0]);
            let heights = (0..).map(|b| 1usize << b).take_while(|&g| g < gm).chain([gm]);
            let orders: Vec<TileOrder> = heights.filter_map(TileOrder::grouped).collect();
            let mut buffers = Buffers::default();
            let plans = orders
                .iter()
                .map(|&order| {
                    let cfg = GemmCfg { order, ..cfg };
                    buffers.plan(
                        gemm_program(&planner, [m, n, k], Epilogue::DEFAULT, cfg)?,
                        &cfg.lowering(&planner.device),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let times = best_of(&plans, rounds);
            let default = TileOrder::DEFAULT.group(gm);
            let base = orders.iter().position(|o| o.group(gm) == default).and_then(|i| times[i]);
            let priced = |order| p.admit(&planner, &GemmCfg { order, ..cfg }).map_or(f64::NAN, |(ns, _)| ns);
            let model = priced(TileOrder::DEFAULT);
            println!("== {m}x{n}x{k} {what}: {} ({gm} tile rows)", gemm_label(&cfg));
            for (order, t) in orders.iter().zip(&times) {
                let mark = match (order.group(gm) == cfg.order.group(gm), order.group(gm) == default) {
                    (true, true) => " planned, default",
                    (true, false) => " planned",
                    (false, true) => " default",
                    _ => "",
                };
                let against = base.zip(*t).map_or(f64::NAN, |(b, t)| t.as_secs_f64() / b.as_secs_f64());
                println!(
                    "  g{:<5} {} x{against:.3}  model x{:.3}{mark}",
                    order.rows(),
                    rate(*t, flops, "TFLOP/s"),
                    priced(*order) / model
                );
            }
        }
    }
    Ok(())
}

/// The probed layers' epilogues beside the plain one: `(label, m, n, k, epilogue)`.
fn fit_gemms() -> Vec<(&'static str, [usize; 3], Epilogue)> {
    let plain = Epilogue::DEFAULT;
    let gelu = Epilogue { act: Act::Gelu, ..plain };
    let residual = Epilogue { residual: true, ..plain };
    let swiglu = Epilogue { act: Act::Silu, gated: true, ..plain };
    vec![
        ("gemm4096", [4096, 4096, 4096], plain),
        ("nemotron_qkv", [704, 1536, 512], plain),
        ("nemotron_ff1", [704, 2048, 512], gelu),
        ("nemotron_ff2", [704, 512, 2048], residual),
        ("qwen3_qkv", [3072, 4096, 1024], plain),
        ("qwen3_gate_up", [3072, 3072, 1024], swiglu),
        ("qwen3_down", [3072, 1024, 3072], residual),
        ("whisper_fc1", [1500, 5120, 1280], gelu),
        ("whisper_fc2", [1500, 1280, 5120], residual),
    ]
}

/// `(label, batch, heads, t, d, causal)`.
const FIT_ATTENTION: [(&str, usize, usize, usize, usize, bool); 6] = [
    ("fa_d64", 4, 8, 2048, 64, false),
    ("fa_d64_causal", 4, 8, 2048, 64, true),
    ("fa_d128", 4, 8, 2048, 128, false),
    ("fa_d128_causal", 4, 8, 2048, 128, true),
    ("whisper_enc_attn", 1, 20, 1500, 64, false),
    ("qwen3_attn", 8, 16, 384, 128, true),
];

/// `(label, rows, depth)`: an f16 NegDot self-join keeping a row's least.
const FIT_NEAREST: [(&str, usize, usize); 4] =
    [("nn_d256", 16384, 256), ("nn_d512", 16384, 512), ("nn_d768", 16384, 768), ("nn_d1024", 16384, 1024)];

/// What compiling a config showed, as the tuner names it.
#[derive(Debug)]
enum Verdict {
    Failed,
    Spilled,
    Clean,
}

/// One shape of the fit: its ranked configs' predicted ns, programs and
/// pick, timed and printed.
struct Fit<'a, C> {
    label: &'a str,
    ranked: Vec<C>,
    predicted: Vec<f64>,
    pick: C,
    tuning: Duration,
}

impl<C: Copy + PartialEq + std::fmt::Debug> Fit<'_, C> {
    fn run(self, picks: bool, rounds: usize, program: impl Fn(C) -> Result<(Program, Lowering)>) {
        let Fit { label, ranked, predicted, pick, tuning } = self;
        println!("tune|{label}|{:.2}", tuning.as_secs_f64());
        let timed: &[C] = if picks { &[] } else { &ranked };
        let mut buffers = Buffers::default();
        let (mut plans, mut verdicts, mut at) = (vec![], vec![], vec![]);
        for &c in timed.iter().chain([&pick]) {
            match program(c).and_then(|(prog, lowering)| buffers.plan(prog, &lowering)) {
                Ok(plan) => {
                    let spilled = plan
                        .kernels()
                        .filter_map(|k| k.program.resource_usage())
                        .any(|r| r.scratch_bytes.is_some_and(|b| b > 0));
                    verdicts.push(if spilled { Verdict::Spilled } else { Verdict::Clean });
                    at.push(Some(plans.len()));
                    plans.push(plan);
                }
                Err(_) => {
                    verdicts.push(Verdict::Failed);
                    at.push(None);
                }
            }
        }
        let times = best_of(&plans, rounds);
        let ns = |i: usize| at[i].and_then(|p| times[p]).map_or("-".into(), |t| t.as_nanos().to_string());
        for (rank, c) in timed.iter().enumerate() {
            println!("fit|{label}|{rank}|{:?}|{:.1}|{}|{c:?}", verdicts[rank], predicted[rank], ns(rank));
        }
        println!("pick|{label}|{}|{pick:?}", ns(timed.len()));
    }
}

/// `f`'s result and its wall time.
fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    (f(), start.elapsed())
}

fn fit_bench(args: &Args) -> Result<()> {
    let (picks, rounds) = (args.flag("--picks"), args.rounds(2)?);
    let only: Option<Vec<&str>> = args.opt("--only").map(|o| o.split(',').collect());
    let wanted = |label: &str| only.as_ref().is_none_or(|o| o.contains(&label));
    let planner = planner()?;
    let store = TuneStore::global();
    println!("costs|{:?}|unmeasured {:?}", planner.device.costs, planner.unmeasured);
    for (label, [m, n, k], epilogue) in fit_gemms().into_iter().filter(|s| wanted(s.0)) {
        let ranked = planner.gemm_ranked(1, m, n, k, &epilogue).to_vec();
        let p = GemmProblem::new(1, m, n, k, &epilogue);
        let predicted = ranked.iter().map(|c| p.admit(&planner, c).map_or(f64::NAN, |a| a.0)).collect();
        let (pick, tuning) = timed(|| gemm_pick(&planner, [m, n, k], epilogue, &ranked));
        Fit { label, ranked, predicted, pick, tuning }
            .run(picks, rounds, |c| Ok((gemm_program(&planner, [m, n, k], epilogue, c)?, c.lowering(&planner.device))));
    }
    for (label, b, h, t, d, causal) in FIT_ATTENTION.into_iter().filter(|s| wanted(s.0)) {
        let ranked = planner.attention_ranked(b * h, t, t, d, causal, false).to_vec();
        let p = AttentionProblem { tiles: b * h, t, tk: t, d, causal, bias: false };
        let predicted = ranked.iter().map(|c| p.admit(&planner, c).map_or(f64::NAN, |a| a.0)).collect();
        let mask = AttnMask { causal, ..AttnMask::default() };
        let spec = AttnSpec {
            batch: Batch::Static(b),
            t,
            tk: t,
            heads: h,
            kv_heads: h,
            d,
            mask,
            cache: None,
            packed: false,
            scale: 1.0 / (d as f32).sqrt(),
        };
        let (pick, tuning) =
            timed(|| tuned::attention(store, &planner, DType::BFloat16, &ranked, spec.clone(), None).cfg());
        Fit { label, ranked, predicted, pick, tuning }.run(picks, rounds, |c| {
            let admitted = spec.clone().admitted(&planner, c).map_err(|why| format!("{c:?}: {why:?}"))?;
            Ok((attention::<BF16>(&admitted), c.lowering(&planner.device)))
        });
    }
    for (label, rows, depth) in FIT_NEAREST.into_iter().filter(|s| wanted(s.0)) {
        let (metric, floors, k) = (Metric::NegDot, Floors { row: false, col: false }, 1);
        let extent = Extent { dims: vec![rows, depth], var: false };
        let dtypes = [DType::Float16, DType::Float16];
        let Plan::Kernel(ranked) =
            shape::nearest(Some(&planner), &dtypes, Some(&extent), Some(&extent), metric, floors, false, k)
        else {
            println!("tune|{label}|- (the op takes the graph)");
            continue;
        };
        let ranked = ranked.to_vec();
        let d = planner.nearest_depth(depth, ScalarDType::Float16).ok_or("no matrix core")?;
        let p = NearestProblem { n: rows, m: rows, d, dtype: ScalarDType::Float16, metric, floors, groups: false, k };
        let predicted = ranked.iter().map(|c| p.admit(&planner, c).map_or(f64::NAN, |a| a.0)).collect();
        let spec = NearestSpec {
            n: rows,
            m: rows,
            d,
            dtype: ScalarDType::Float16,
            metric,
            floors,
            groups: false,
            k,
            live: Default::default(),
        };
        let (pick, tuning) = timed(|| tuned::nearest(store, &planner, DType::Float16, &ranked, spec.clone()).cfg());
        Fit { label, ranked, predicted, pick, tuning }.run(picks, rounds, |c| {
            let admitted = spec.clone().admitted(&planner, c).map_err(|why| format!("{c:?}: {why:?}"))?;
            Ok((nearest::<F16>(&admitted), Lowering::new(&planner.device)))
        });
    }
    Ok(())
}

fn trips_bench() -> Result<()> {
    let target = planner()?.device.target.clone();
    for (op, unroll, stages) in [("probe:trip", true, 2), ("probe:trip_rolled", false, 2), ("probe:trip_one", false, 1)]
    {
        for per_sm in [1usize, 2, 3, 4, 6, 8, 12, 24, 48] {
            match svod_tk::probe::trip_at(&target, op, unroll, stages, per_sm)? {
                Some(t) => println!(
                    "residency|{op}|{per_sm}|trip {:.3}|wall {:.3}|spread {:.3}|base {:.3}|measurable {}",
                    t.trip,
                    t.trip * per_sm as f64,
                    t.spread,
                    t.base,
                    t.measurable,
                ),
                None => println!("residency|{op}|{per_sm}|unresolved"),
            }
        }
    }
    Ok(())
}

/// A pinned config, `BMxBNxBK:S:WRxWC[:rolled]`, on the warp core in the
/// default raster.
fn pinned(code: &str) -> Result<GemmCfg> {
    let dims = |s: &str| -> Result<Vec<usize>> { Ok(s.split('x').map(str::parse).collect::<Result<_, _>>()?) };
    let parts: Vec<&str> = code.split(':').collect();
    let (tile, stages, warps) = match parts[..] {
        [tile, stages, warps] | [tile, stages, warps, "rolled"] => (dims(tile)?, stages.parse()?, dims(warps)?),
        _ => return Err(format!("{code}: BMxBNxBK:S:WRxWC[:rolled]").into()),
    };
    let (&[bm, bn, bk], &[wr, wc]) = (&tile[..], &warps[..]) else {
        return Err(format!("{code}: three tile dims, two warp dims").into());
    };
    Ok(GemmCfg {
        tile: [bm, bn, bk],
        warps: [wr as u32, wc as u32],
        core: Core::Warp,
        schedule: Schedule::ring(stages, parts.len() == 3),
        order: TileOrder::DEFAULT,
        tail: None,
    })
}

fn pin_bench(args: &Args) -> Result<()> {
    let rounds = args.rounds(20)?;
    let planner = planner()?;
    let gated = args.flag("--gated");
    let act = if gated { Act::Silu } else { Act::None };
    let epilogue = Epilogue { residual: args.flag("--residual"), gated, act, ..Epilogue::DEFAULT };
    let cfgs = args.opt("--cfg").ok_or("--cfg BMxBNxBK:S:WRxWC[:rolled],…")?.split(',').map(pinned);
    let cfgs = cfgs.collect::<Result<Vec<_>>>()?;
    let (target, costs) = (&planner.device.target, planner.device.costs);
    let zeroed = [
        ("latency", Costs { latency: 0.0, latency_one: 0.0, ..costs }),
        ("lone trip", Costs { trip_lone: 0.0, trip_one_lone: 0.0, ..costs }),
        ("step", Costs { step: 0.0, ..costs }),
    ];
    let zeroed =
        zeroed.map(|(term, costs)| (term, Planner::new(Device { target: target.clone(), costs }).expect("SMs")));
    println!("costs|{costs:?}");
    for [m, n, k] in args.shapes::<3>("1500x1280x5120")? {
        let p = GemmProblem::new(1, m, n, k, &epilogue);
        let mut buffers = Buffers::default();
        let plans = cfgs
            .iter()
            .map(|&c| buffers.plan(gemm_program(&planner, [m, n, k], epilogue, c)?, &c.lowering(&planner.device)))
            .collect::<Result<Vec<_>>>()?;
        for (c, t) in cfgs.iter().zip(best_of(&plans, rounds)) {
            let priced = |planner: &Planner| p.admit(planner, c).map_or(f64::NAN, |a| a.0);
            let blocks = m.div_ceil(c.tile[0]) * n.div_ceil(c.tile[1]);
            let less: Vec<String> = zeroed.iter().map(|(term, z)| format!("less {term} {:.0}", priced(z))).collect();
            println!(
                "pin|{m}x{n}x{k}|{}|blocks {blocks}|measured {}|model {:.0}|{}",
                gemm_label(c),
                t.map_or("failed".into(), |t| t.as_nanos().to_string()),
                priced(&planner),
                less.join("|")
            );
        }
    }
    Ok(())
}
