//! Device benchmarks of tk's kernels on the default device (`SVOD_DEVICE`),
//! every time as the tuner times its candidates ([`tune::measure`]): a
//! config's programs (a split's merge, a tail's partials included) and an
//! op's plan alike, on the device's own clock.
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
//! - `sqtt gemm M N K [--cfg I|tuned|BMxBNxBK:S:WRxWC[:rolled],…] [--top N]
//!   [--out DIR] [--json] [--attribute] [--residual] [--gated]`: each config
//!   (the tuner's pick by default, `I` the planner's `I`th) timed untraced,
//!   then traced once on an AMD RDNA3/RDNA4 card (SQTT), its trace saved
//!   under `DIR` (`./sqtt`) with its tile statements where a host build with
//!   a line table compiles to the traced code (`kernel.tk`, `lines.json`;
//!   else the `lines=mismatch` label says why), and its report printed (`--top` rows a table,
//!   `--json` one line); `--attribute` sets the model's terms (`price` less
//!   each) beside the traced SIMD's cycles of their counterparts.
//!   `sqtt attention B H T D [--causal] [--bias] [--cfg I|tuned,…]` traces
//!   attention over `[B, T, H, D]` bf16 the same way; `--flip FORM,…` flips
//!   those softmax forms (`Forms` fields) of each picked config, `--ring S`
//!   sets its ring's slots.
//!   `SVOD_SQTT_PSTATE=standard|peak` pins the traced pass's clock.
//!   `SVOD_SQTT_SIMD=S[,S...]` picks each SE's traced SIMD (default `0,1`).
//!   `sqtt --load DIR [--json] [--attribute]` re-reads a saved trace, no GPU;
//!   `sqtt --compare DIR_A DIR_B [--json] [--top N]` sets B's trace against
//!   A's ([`Diff`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use svod_dtype::default_device::default_device;
use svod_dtype::{DType, ScalarDType};
use svod_runtime::sqtt::{Analysis, Capture, Class, Diff};
use svod_runtime::{ExecutionPlan, ProfileOptions, Raw, Sink, SqttOptions};
use svod_tensor::Tensor;
use svod_tk::atoms::{Core, Costs, Device};
use svod_tk::ir::Schedule;
use svod_tk::kernels::attention::{AttnMask, AttnSpec, Bias, FaCfg};
use svod_tk::kernels::conv::{ConvCfg, ConvGeom, ConvSpec};
use svod_tk::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, TileOrder};
use svod_tk::kernels::nearest::{Floors, Metric, NearestCfg, NearestSpec};
use svod_tk::kernels::{Act, Batch};
use svod_tk::ops::config::{AttentionProblem, ConvProblem, GemmProblem, NearestProblem, Planner};
use svod_tk::ops::shape::{self, Extent, Plan};
use svod_tk::ops::{self, Attn, Nearest, Quantized, Rerank, Search, candidates, tuned};
use svod_tk::tune::{self, Candidate, Compiled, TuneStore};

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

const USAGE: &str =
    "tk_bench <gemm|attention|nearest|rerank|raster|fit|trips|pin|sqtt> [options]; see the example's docs";

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
        "sqtt" => sqtt_bench(&args),
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
}

/// The op layer's planner of the default device: its cost probes measured
/// (or read from the tune store) once.
fn planner() -> Result<Arc<Planner>> {
    ops::planner(&default_device()).ok_or_else(|| "no GPU tk has kernels for: set SVOD_DEVICE".into())
}

/// Each candidate's best device time, as the tuner times it.
fn timed_candidates(candidates: Vec<Candidate>) -> Vec<(Compiled, Option<Duration>)> {
    tune::measure(candidates).into_iter().map(|(verdict, ns)| (verdict, ns.map(Duration::from_nanos))).collect()
}

/// Each op plan's best device time, timed as a candidate is.
fn timed_plans(plans: Vec<ExecutionPlan>) -> Vec<Option<Duration>> {
    tune::measure_plans(plans).into_iter().map(|ns| ns.map(Duration::from_nanos)).collect()
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

/// A bf16 GEMM at `cfg`, admitted: the programs the op launches.
fn gemm_candidate(planner: &Planner, [m, n, k]: [usize; 3], epilogue: Epilogue, cfg: GemmCfg) -> Result<Candidate> {
    let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue };
    let spec = spec.admitted(planner, cfg).map_err(|why| format!("{cfg:?}: {why:?}"))?;
    Ok(candidates::gemm(&spec, &DType::BFloat16, planner))
}

/// The tuner's GEMM pick for `shape`, as the device tests and GEMM probes
/// tune ([`tuned::gemm_probe`]).
fn gemm_pick(planner: &Planner, [m, n, k]: [usize; 3], epilogue: Epilogue, ranked: &[GemmCfg]) -> GemmCfg {
    let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue };
    tuned::gemm_probe(TuneStore::global(), planner, DType::BFloat16, ranked, spec).cfg()
}

fn gemm_bench(args: &Args) -> Result<()> {
    let top = args.num("--top", 0)?;
    let planner = planner()?;
    for [m, n, k] in args.shapes::<3>("4096x4096x4096")? {
        let ranked = planner.gemm_ranked(1, m, n, k, &Epilogue::DEFAULT);
        let pick = gemm_pick(&planner, [m, n, k], Epilogue::DEFAULT, &ranked);
        let mut cfgs: Vec<GemmCfg> = ranked.iter().take(top).copied().collect();
        if !cfgs.contains(&pick) {
            cfgs.push(pick);
        }
        let candidates = cfgs
            .iter()
            .map(|&c| gemm_candidate(&planner, [m, n, k], Epilogue::DEFAULT, c))
            .collect::<Result<Vec<_>>>()?;
        println!("== gemm {m}x{n}x{k} bf16");
        for (c, (_, t)) in cfgs.iter().zip(timed_candidates(candidates)) {
            let mark = if *c == pick { " (tuner's pick)" } else { "" };
            println!("  {}: {}{mark}", gemm_label(c), rate(t, 2.0 * (m * n * k) as f64, "TFLOP/s"));
        }
    }
    Ok(())
}

fn attention_bench(args: &Args) -> Result<()> {
    let top = args.num("--top", 0)?;
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
        let mut times = timed_plans(vec![o.prepare()?]);
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
        let mut ranked = vec![];
        for &c in planner.attention_ranked(b * h, t, t, d, causal, bias).iter().take(top) {
            let admitted = spec.clone().admitted(&planner, c).map_err(|why| format!("{c:?}: {why:?}"))?;
            ranked.push(candidates::attention(&admitted, &[t], &DType::BFloat16, &planner));
            labels.push(format!("ranked {c:?}"));
        }
        times.extend(timed_candidates(ranked).into_iter().map(|(_, t)| t));
        let flops = 4.0 * (b * h * t * t * d) as f64 / if causal { 2.0 } else { 1.0 };
        println!("== attention b{b} h{h} t{t} d{d} causal={causal} bias={bias}");
        for (label, time) in labels.iter().zip(times) {
            println!("  {}: {label}", rate(time, flops, "TFLOP/s"));
        }
    }
    Ok(())
}

fn nearest_bench(args: &Args) -> Result<()> {
    let (n, k) = (args.num("--rows", 1 << 16)?, args.num("--k", 1)?);
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
            let [time] = timed_plans(vec![index.prepare()?])[..] else { unreachable!("one plan") };
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
    let n = args.num("--rows", 16384)?;
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
            let times = timed_plans(plans);
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
            let candidates = orders
                .iter()
                .map(|&order| gemm_candidate(&planner, [m, n, k], Epilogue::DEFAULT, GemmCfg { order, ..cfg }))
                .collect::<Result<Vec<_>>>()?;
            let times: Vec<_> = timed_candidates(candidates).into_iter().map(|(_, t)| t).collect();
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
    fn run(self, picks: bool, candidate: impl Fn(C) -> Result<Candidate>) {
        let Fit { label, ranked, predicted, pick, tuning } = self;
        println!("tune|{label}|{:.2}", tuning.as_secs_f64());
        let timed: &[C] = if picks { &[] } else { &ranked };
        let (built, at): (Vec<Candidate>, Vec<Option<usize>>) = {
            let (mut built, mut at) = (vec![], vec![]);
            for &c in timed.iter().chain([&pick]) {
                at.push(candidate(c).ok().map(|programs| {
                    built.push(programs);
                    built.len() - 1
                }));
            }
            (built, at)
        };
        let times = timed_candidates(built);
        let verdict = |i: usize| at[i].map_or(Compiled::Failed, |b| times[b].0);
        let ns = |i: usize| at[i].and_then(|b| times[b].1).map_or("-".into(), |t| t.as_nanos().to_string());
        for (rank, c) in timed.iter().enumerate() {
            println!("fit|{label}|{rank}|{:?}|{:.1}|{}|{c:?}", verdict(rank), predicted[rank], ns(rank));
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
    let picks = args.flag("--picks");
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
        Fit { label, ranked, predicted, pick, tuning }.run(picks, |c| gemm_candidate(&planner, [m, n, k], epilogue, c));
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
        Fit { label, ranked, predicted, pick, tuning }.run(picks, |c| {
            let admitted = spec.clone().admitted(&planner, c).map_err(|why| format!("{c:?}: {why:?}"))?;
            Ok(candidates::attention(&admitted, &[t], &DType::BFloat16, &planner))
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
        Fit { label, ranked, predicted, pick, tuning }.run(picks, |c| {
            let admitted = spec.clone().admitted(&planner, c).map_err(|why| format!("{c:?}: {why:?}"))?;
            Ok(candidates::nearest(&planner, DType::Float16, &admitted))
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

/// `--residual` and `--gated` (a SwiGLU) on the plain epilogue.
fn epilogue(args: &Args) -> Epilogue {
    let gated = args.flag("--gated");
    let act = if gated { Act::Silu } else { Act::None };
    Epilogue { residual: args.flag("--residual"), gated, act, ..Epilogue::DEFAULT }
}

fn pin_bench(args: &Args) -> Result<()> {
    let planner = planner()?;
    let epilogue = epilogue(args);
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
        let candidates =
            cfgs.iter().map(|&c| gemm_candidate(&planner, [m, n, k], epilogue, c)).collect::<Result<Vec<_>>>()?;
        for (c, (_, t)) in cfgs.iter().zip(timed_candidates(candidates)) {
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

/// The terms `pin` prices by: each with its costs zeroed.
fn zeroed_terms(costs: Costs) -> [(&'static str, Costs); 6] {
    [
        ("steps", Costs { step: 0.0, step_group: 0.0, step_wide: 0.0, step_i8: 0.0, ..costs }),
        ("reads", Costs { smem_byte: 0.0, ..costs }),
        ("fill", Costs { global_byte: 0.0, l2_byte: 0.0, l2_bytes: 0.0, ..costs }),
        ("wait", Costs { latency: 0.0, latency_one: 0.0, ..costs }),
        ("lone_trip", Costs { trip_lone: 0.0, trip_one_lone: 0.0, ..costs }),
        ("trip", Costs { trip: 0.0, trip_rolled: 0.0, trip_one: 0.0, ..costs }),
    ]
}

/// The labels a GEMM trace carries for `--attribute`: the model's price of
/// `cfg`, the price less each term, the trips a block runs and the blocks
/// the busiest SM runs.
fn gemm_model(planner: &Planner, [m, n, k]: [usize; 3], epilogue: Epilogue, cfg: GemmCfg) -> BTreeMap<String, String> {
    let p = GemmProblem::new(1, m, n, k, &epilogue);
    let priced = |planner: &Planner| p.admit(planner, &cfg).map_or(f64::NAN, |a| a.0);
    let model = priced(planner);
    let target = &planner.device.target;
    let mut labels = BTreeMap::from([
        ("op".into(), "gemm".into()),
        ("shape".into(), format!("{m}x{n}x{k}")),
        ("cfg".into(), gemm_label(&cfg)),
        ("model.ns".into(), format!("{model:.0}")),
        ("model.trips".into(), k.div_ceil(cfg.tile[2]).to_string()),
    ]);
    if let Some(sms) = target.sms {
        let blocks = m.div_ceil(cfg.tile[0]) * n.div_ceil(cfg.tile[1]);
        labels.insert("model.blocks_per_sm".into(), blocks.div_ceil(sms as usize).to_string());
    }
    for (term, costs) in zeroed_terms(planner.device.costs) {
        let less = Planner::new(Device { target: target.clone(), costs }).map_or(f64::NAN, |z| priced(&z));
        labels.insert(format!("model.{term}"), format!("{:.0}", model - less));
    }
    labels
}

fn sqtt_bench(args: &Args) -> Result<()> {
    let top = args.num("--top", 12)?;
    if let Some(dir) = args.opt("--load") {
        return sqtt_print(&Capture::load(Path::new(dir))?, args, top);
    }
    if let Some(at) = args.0.iter().position(|a| a == "--compare") {
        let [a, b] = [1, 2].map(|i| args.0.get(at + i).filter(|d| !d.starts_with("--")));
        let (Some(a), Some(b)) = (a, b) else { return Err("sqtt --compare DIR_A DIR_B [--json] [--top N]".into()) };
        let diff = Diff::load(Path::new(a), Path::new(b))?;
        match args.flag("--json") {
            true => println!("{}", diff.to_json()),
            false => print!("{}", diff.text(top)),
        }
        return Ok(());
    }
    let positional: Vec<&str> = args.0.iter().map(String::as_str).take_while(|a| !a.starts_with("--")).collect();
    let usage = "sqtt gemm M N K [--cfg I|tuned|BMxBNxBK:S:WRxWC[:rolled],…], \
                 sqtt attention B H T D [--causal] [--bias] [--cfg I|tuned,…], \
                 sqtt nearest N D K [--metric M] [--cfg I|tuned|ranked,…] [--splits S], \
                 sqtt conv H W CIN COUT KERNEL STRIDE [--batch B] [--residual] [--cfg I|tuned|ranked,…], \
                 sqtt --load DIR or sqtt --compare A B";
    let planner = planner()?;
    let traced: Vec<Traced> = match positional[..] {
        ["gemm", m, n, k] => gemm_traced(&planner, [m.parse()?, n.parse()?, k.parse()?], args)?,
        ["attention", b, h, t, d] => {
            attention_traced(&planner, [b.parse()?, h.parse()?, t.parse()?, d.parse()?], args)?
        }
        ["nearest", n, d, k] => nearest_traced(&planner, [n.parse()?, d.parse()?, k.parse()?], args)?,
        ["conv", h, w, cin, cout, kernel, stride] => {
            let dims = [h, w, cin, cout, kernel, stride].map(str::parse::<usize>);
            conv_traced(&planner, dims.into_iter().collect::<Result<Vec<_>, _>>()?.try_into().expect("six"), args)?
        }
        _ => return Err(usage.into()),
    };
    let out = PathBuf::from(args.opt("--out").unwrap_or("sqtt"));
    for Traced { name, candidate, labels } in traced {
        let untraced = timed_candidates(vec![candidate()?]).pop().and_then(|t| t.1);
        let dir = out.join(&name);
        // `SVOD_SQTT_PSTATE` pins the traced pass's clock, `SVOD_SQTT_SIMD` picks its SIMDs.
        let sqtt = SqttOptions { config: SqttOptions::config_from_env()?, sink: Sink::Disk(dir.clone()) };
        let opts = ProfileOptions { iters: 3, static_analysis: false, sqtt: Some(sqtt), ..ProfileOptions::from_env() };
        let profiles = tune::profile(candidate()?, &opts).map_err(|e| format!("{name}: {e}"))?;
        // `SVOD_PMC` adds the hardware counters of the untraced passes.
        if opts.counters != svod_runtime::profiler::PmcSelection::None {
            profiles.iter().for_each(|p| print!("{}", p.render_table()));
        }
        let traces: Vec<_> = profiles.iter().flat_map(|p| &p.stages).flat_map(|s| &s.sqtt).collect();
        if traces.is_empty() {
            return Err("no thread trace: SQTT needs an AMD RDNA3/RDNA4 device (SVOD_DEVICE=AMD)".into());
        }
        let candidate = candidate()?;
        for trace in traces {
            let Raw::Saved(dir) = &trace.raw else { unreachable!("a disk sink saves") };
            if let Err(error) = &trace.analysis {
                return Err(format!("{}: {error}", dir.display()).into());
            }
            let mut capture = Capture::load(dir)?;
            capture.labels.extend(labels.clone());
            capture.labels.insert("candidate_ns".into(), untraced.map_or("-".into(), |t| t.as_nanos().to_string()));
            // The statements, from a host build with a line table whose code is the traced one.
            match tune::twin(&candidate, &mut capture) {
                None => capture.labels.insert("lines".into(), "twin".into()),
                Some(why) => {
                    eprintln!("{}: no statements: {why}", dir.display());
                    capture.labels.insert("lines.why".into(), why);
                    capture.labels.insert("lines".into(), "mismatch".into())
                }
            };
            capture.save(dir)?;
            println!("saved {}", dir.display());
            sqtt_print(&capture, args, top)?;
        }
    }
    Ok(())
}

/// One configuration `sqtt` traces: its capture directory, its candidate
/// (built afresh for each pass) and the labels its capture carries.
struct Traced<'a> {
    name: String,
    candidate: Box<dyn Fn() -> Result<Candidate> + 'a>,
    labels: BTreeMap<String, String>,
}

fn gemm_traced<'a>(planner: &'a Planner, shape: [usize; 3], args: &Args) -> Result<Vec<Traced<'a>>> {
    let [m, n, k] = shape;
    let epilogue = epilogue(args);
    let ranked = planner.gemm_ranked(1, m, n, k, &epilogue);
    let cfgs = args.opt("--cfg").unwrap_or("tuned").split(',').map(|c| match (c, c.parse::<usize>()) {
        ("tuned", _) => Ok(gemm_pick(planner, shape, epilogue, &ranked)),
        (_, Ok(i)) => ranked.get(i).copied().ok_or_else(|| format!("--cfg {i}: {} ranked", ranked.len()).into()),
        (code, _) => pinned(code),
    });
    cfgs.enumerate()
        .map(|(i, cfg)| {
            let cfg = cfg?;
            Ok(Traced {
                name: format!("gemm_{m}x{n}x{k}_{i}"),
                candidate: Box::new(move || gemm_candidate(planner, shape, epilogue, cfg)),
                labels: gemm_model(planner, shape, epilogue, cfg),
            })
        })
        .collect()
}

/// Attention over `[B, T, H, D]` bf16 operands; `--cfg tuned` (the default)
/// is the configuration the op launches, an index one of the planner's ranking.
fn attention_traced<'a>(planner: &'a Planner, [b, h, t, d]: [usize; 4], args: &Args) -> Result<Vec<Traced<'a>>> {
    let (causal, bias) = (args.flag("--causal"), args.flag("--bias"));
    let ranked = planner.attention_ranked(b * h, t, t, d, causal, bias);
    let spec = AttnSpec {
        batch: Batch::Static(b),
        t,
        tk: t,
        heads: h,
        kv_heads: h,
        d,
        mask: AttnMask { causal, bias: bias.then_some(Bias::PerBatch), ..AttnMask::default() },
        cache: None,
        packed: false,
        scale: 1.0 / (d as f32).sqrt(),
    };
    let picks = args.opt("--cfg").unwrap_or("tuned").split(',').map(|c| match c {
        "tuned" => Ok((
            "t".to_string(),
            tuned::attention(TuneStore::global(), planner, DType::BFloat16, &ranked, spec.clone(), None),
        )),
        i => {
            let i = i.parse::<usize>()?;
            let cfg = *ranked.get(i).ok_or_else(|| format!("--cfg {i}: {} ranked", ranked.len()))?;
            Ok((i.to_string(), spec.clone().admitted(planner, cfg).map_err(|why| format!("{cfg:?}: {why:?}"))?))
        }
    });
    picks
        .map(|pick: Result<_>| {
            let (mut i, mut admitted) = pick?;
            if let Some(flips) = args.opt("--flip") {
                let mut cfg = admitted.cfg();
                for name in flips.split(',') {
                    let f = &mut cfg.forms;
                    let form = match name {
                        "truncate_p" => &mut f.truncate_p,
                        "sum_after" => &mut f.sum_after,
                        "skip_rescale" => &mut f.skip_rescale,
                        "raw_max" => &mut f.raw_max,
                        "bias_direct" => &mut f.bias_direct,
                        other => return Err(format!("--flip {other}: a Forms field").into()),
                    };
                    *form = !*form;
                }
                admitted = spec.clone().admitted(planner, cfg).map_err(|why| format!("{cfg:?}: {why:?}"))?;
                i = format!("{i}_{}", flips.replace(',', "+"));
            }
            if let Some(depth) = args.opt("--ring") {
                let cfg = admitted.cfg();
                let cfg = FaCfg { schedule: Schedule::ring(depth.parse()?, cfg.unroll()), ..cfg };
                admitted = spec.clone().admitted(planner, cfg).map_err(|why| format!("{cfg:?}: {why:?}"))?;
                i = format!("{i}_ring{depth}");
            }
            let cfg = admitted.cfg();
            let labels = BTreeMap::from([
                ("op".into(), "attention".into()),
                ("shape".into(), format!("b{b} h{h} t{t} d{d} causal={causal} bias={bias}")),
                ("cfg".into(), format!("{cfg:?}")),
            ]);
            Ok(Traced {
                name: format!(
                    "attention_b{b}h{h}t{t}d{d}{}{}_{i}",
                    ["", "c"][causal as usize],
                    ["", "b"][bias as usize]
                ),
                candidate: Box::new(move || Ok(candidates::attention(&admitted, &[t], &DType::BFloat16, planner))),
                labels,
            })
        })
        .collect()
}

/// A self-join of `N` f16 rows of depth `D` keeping `K` a row (`--metric`,
/// NegDot by default); `--cfg tuned` (the default) is the op's pick, an index
/// one of the shape gate's ranking.
fn nearest_traced<'a>(planner: &'a Planner, [n, depth, k]: [usize; 3], args: &Args) -> Result<Vec<Traced<'a>>> {
    let (metric, floors) = (args.metric()?.unwrap_or(Metric::NegDot), Floors { row: false, col: false });
    let extent = Extent { dims: vec![n, depth], var: false };
    let Plan::Kernel(ranked) =
        shape::nearest(Some(planner), &[DType::Float16; 2], Some(&extent), Some(&extent), metric, floors, false, k)
    else {
        return Err("nearest: the op takes the graph at this shape".into());
    };
    let d = planner.nearest_depth(depth, ScalarDType::Float16).ok_or("no matrix core")?;
    let spec = NearestSpec {
        n,
        m: n,
        d,
        dtype: ScalarDType::Float16,
        metric,
        floors,
        groups: false,
        k,
        live: Default::default(),
    };
    if args.opt("--cfg") == Some("ranked") {
        let p = NearestProblem { n, m: n, d, dtype: ScalarDType::Float16, metric, floors, groups: false, k };
        for (i, cfg) in ranked.iter().enumerate() {
            println!("{i:4} {:12.0} {cfg:?}", p.admit(planner, cfg).map_or(f64::NAN, |a| a.0));
        }
        return Ok(vec![]);
    }
    let picks = args.opt("--cfg").unwrap_or("tuned").split(',').map(|c| match c {
        "tuned" => {
            Ok(("t".to_string(), tuned::nearest(TuneStore::global(), planner, DType::Float16, &ranked, spec.clone())))
        }
        i => {
            let i = i.parse::<usize>()?;
            let cfg = *ranked.get(i).ok_or_else(|| format!("--cfg {i}: {} ranked", ranked.len()))?;
            Ok((i.to_string(), spec.clone().admitted(planner, cfg).map_err(|why| format!("{cfg:?}: {why:?}"))?))
        }
    });
    picks
        .map(|pick: Result<_>| {
            let (mut i, mut admitted) = pick?;
            if let Some(splits) = args.opt("--splits") {
                let cfg = NearestCfg { splits: splits.parse()?, ..admitted.cfg() };
                admitted = spec.clone().admitted(planner, cfg).map_err(|why| format!("{cfg:?}: {why:?}"))?;
                i = format!("{i}_s{splits}");
            }
            let labels = BTreeMap::from([
                ("op".into(), "nearest".into()),
                ("shape".into(), format!("n{n} d{depth} k{k} {metric:?}")),
                ("cfg".into(), format!("{:?}", admitted.cfg())),
            ]);
            Ok(Traced {
                name: format!("nearest_n{n}d{depth}k{k}{metric:?}_{i}"),
                candidate: Box::new(move || Ok(candidates::nearest(planner, DType::Float16, &admitted))),
                labels,
            })
        })
        .collect()
}

/// A YOLO block's f16 convolution over `[B, H, W, CIN]`, padded to keep the
/// size at stride 1, with bias and SiLU in the epilogue (`--residual` adds
/// one); `--cfg tuned` (the default) is the op's pick, an index one of the
/// planner's ranking under its split, `ranked` lists the ranking.
fn conv_traced<'a>(
    planner: &'a Planner,
    [h, w, cin, cout, kernel, stride]: [usize; 6],
    args: &Args,
) -> Result<Vec<Traced<'a>>> {
    let images = args.num("--batch", 1)?;
    let geom =
        ConvGeom { h, w, cin, cout, kernel: [kernel; 2], stride: [stride; 2], pad: [kernel / 2; 2], dilation: [1; 2] };
    let epilogue = Epilogue { bias: true, act: Act::Silu, residual: args.flag("--residual"), ..Epilogue::DEFAULT };
    let extent = Extent { dims: vec![images], var: false };
    let residual = epilogue.residual.then_some(DType::Float16);
    let plan = shape::conv2d(
        Some(planner),
        &[DType::Float16; 3],
        DType::Float16,
        residual,
        Some(&extent),
        &geom,
        1,
        &epilogue,
    );
    let Plan::Kernel(ranked) = plan else { return Err("conv: the op takes the graph at this shape".into()) };
    let spec = ConvSpec { batch: Batch::Static(images), geom, epilogue, split: 1 };
    if args.opt("--cfg") == Some("ranked") {
        let (batches, m) = shape::conv_grid(&extent, &geom);
        let p = ConvProblem::new(batches, m, &geom, &epilogue);
        for (i, cfg) in ranked.iter().enumerate() {
            println!("{i:4} {:12.0} {cfg:?}", p.admit(planner, &cfg.gemm).map_or(f64::NAN, |a| a.0));
        }
        return Ok(vec![]);
    }
    let shape = format!("{h}x{w}x{cin}>{cout} k{kernel}s{stride} b{images}");
    let picks = args.opt("--cfg").unwrap_or("tuned").split(',').map(|c| match c {
        "tuned" => {
            Ok(("t".to_string(), tuned::conv(TuneStore::global(), planner, DType::Float16, &ranked, spec.clone())))
        }
        i => {
            let i = i.parse::<usize>()?;
            let cfg = *ranked.get(i).ok_or_else(|| format!("--cfg {i}: {} ranked", ranked.len()))?;
            let admitted = ConvSpec { split: cfg.split, ..spec.clone() }.admitted(planner, cfg.gemm);
            Ok((i.to_string(), admitted.map_err(|why| format!("{cfg:?}: {why:?}"))?))
        }
    });
    picks
        .map(|pick: Result<_>| {
            let (i, admitted) = pick?;
            let cfg = ConvCfg { gemm: admitted.cfg(), split: admitted.split };
            let labels = BTreeMap::from([
                ("op".into(), "conv".into()),
                ("shape".into(), shape.clone()),
                ("cfg".into(), format!("{cfg:?}")),
            ]);
            Ok(Traced {
                name: format!("conv_{h}x{w}x{cin}x{cout}k{kernel}s{stride}b{images}_{i}"),
                candidate: Box::new(move || Ok(candidates::conv(&admitted, &DType::Float16, planner))),
                labels,
            })
        })
        .collect()
}

fn sqtt_print(capture: &Capture, args: &Args, top: usize) -> Result<()> {
    let a = capture.analyze()?;
    match args.flag("--json") {
        true => println!("{}", a.to_json()),
        false => print!("{}", a.text(top)),
    }
    if args.flag("--attribute") {
        print!("{}", attribution(&a)?);
    }
    Ok(())
}

/// The model's terms beside the traced SIMD's cycles of what they price:
/// steps the matrix ops' issue less their stall, reads the LDS ops and
/// their waits, fill the global memory ops and their waits, wait the
/// barriers and messages.
fn attribution(a: &Analysis) -> Result<String> {
    use std::fmt::Write;
    let label = |key: &str| a.labels.get(key).and_then(|v| v.parse::<f64>().ok());
    let model =
        label("model.ns").ok_or("no model terms in the trace's labels: a `tk_bench sqtt gemm` capture has them")?;
    let total: i64 = a.classes.iter().map(|c| c.latency).sum::<i64>().max(1);
    let waves = a.complete.max(1) as f64;
    let ns = |cycles: f64| a.sclk_mhz.map_or("-".into(), |mhz| format!("{:.0}", cycles * 1000.0 / mhz as f64));
    let mut o = String::new();
    let sclk = a.sclk_mhz.map_or("-".into(), |mhz| mhz.to_string());
    writeln!(
        o,
        "== attribution (model: ns of the busiest SM, price less the term, terms overlap; measured: traced SIMD cycles a wave (mean of {waves}), ns at the traced sclk {sclk} MHz)"
    )?;
    writeln!(
        o,
        "{:<10} {:>10} {:>7} {:>12} {:>9} {:>11}  counterpart",
        "term", "model ns", "model%", "cycles/wave", "cycles%", "ns at sclk"
    )?;
    let (matrix, matrix_stall) = a.cycles(&[Class::Matrix]);
    let measured: [(&str, Option<(i64, &str)>); 6] = [
        ("steps", Some((matrix - matrix_stall, "matrix issue less stall"))),
        ("reads", Some((a.cycles(&[Class::Lds, Class::WaitLgkm]).0, "lds + wait_lgkm"))),
        ("fill", Some((a.cycles(&[Class::VmemLoad, Class::VmemStore, Class::WaitVmem]).0, "vmem + wait_vmem"))),
        ("wait", Some((a.cycles(&[Class::Barrier, Class::Message]).0, "barrier + message"))),
        ("lone_trip", None),
        ("trip", None),
    ];
    for (term, counterpart) in measured {
        let m = label(&format!("model.{term}")).unwrap_or(f64::NAN);
        let (cycles, share, at, what) = match counterpart {
            Some((c, what)) => {
                let per_wave = c as f64 / waves;
                (format!("{per_wave:.0}"), format!("{:.1}", 100.0 * c as f64 / total as f64), ns(per_wave), what)
            }
            None => ("-".into(), "-".into(), "-".into(), "none traced"),
        };
        writeln!(o, "{term:<10} {m:>10.0} {:>7.1} {cycles:>12} {share:>9} {at:>11}  {what}", 100.0 * m / model)?;
    }
    writeln!(
        o,
        "{:<10} {model:>10.0} {:>7.1} {:>12.0} {:>9.1} {:>11}  all traced instructions",
        "total",
        100.0,
        total as f64 / waves,
        100.0,
        ns(total as f64 / waves)
    )?;
    let trips = label("model.trips").unwrap_or(f64::NAN);
    let per_sm = label("model.blocks_per_sm").unwrap_or(f64::NAN);
    let main_loop = a.main_loop.as_ref();
    let iteration = main_loop.and_then(|l| l.cycles).map(|c| c.median as f64);
    let unroll = main_loop.filter(|l| l.trips.labelled).map(|l| f64::from(l.trips.per_iteration));
    let per_trip = iteration.zip(unroll).map(|(c, u)| c / u);
    writeln!(
        o,
        "per trip   model {:.0} ns over {trips} trips × {per_sm} blocks/SM ({:.1} ns a block-trip); measured {} cycles a wave's iteration (median) of {} trips, {} cycles a trip, {} ns at the traced sclk",
        model / trips,
        model / trips / per_sm,
        iteration.map_or("-".into(), |c| format!("{c:.0}")),
        unroll.map_or("-".into(), |u| format!("{u:.0}")),
        per_trip.map_or("-".into(), |c| format!("{c:.0}")),
        per_trip.map_or("-".into(), ns),
    )?;
    let untraced = a.labels.get("candidate_ns").map_or("-", String::as_str);
    writeln!(
        o,
        "time       model {model:.0} ns, candidate {untraced} ns untraced (never compare the traced pass's ns)"
    )?;
    Ok(o)
}
