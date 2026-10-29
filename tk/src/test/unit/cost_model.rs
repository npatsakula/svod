//! The planner's cost model against the device: every config a ranking
//! admits for a few GEMM and attention shapes (4096³, model projections,
//! flash attention at d64/d128 with and without a causal mask) compiled and
//! timed beside the ns the model ranks it at, and the tuner's pick timed
//! with them. One `fit|…` line a config and a `pick|…` line a shape, for an
//! offline fit of rank correlation and regret; never asserts. And the
//! model's terms on the host, from fixture costs.

use proptest::prelude::*;

use crate::atoms::{Costs, Device, Target};
use crate::build::BF16;
use crate::kernels::attention::{AttnMask, AttnSpec, FaCfg, attention};
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::kernels::{Act, Batch};
use crate::ops::config::{AttentionProblem, DIAGONAL, GemmProblem, Planner, gemm_cfg};
use crate::tune::{Bench, Candidate, Compiled, Pass, TuneStore};

/// Each of `ranked` compiled (in ranked batches, as the tuner admits), then
/// every one that compiled timed twice, its lesser time kept: the verdicts
/// and ns in `ranked`'s order, and the extra candidates' ns after them.
fn timed<C: Copy + PartialEq + Eq + std::hash::Hash + std::fmt::Debug + Send>(
    ranked: &[C],
    extra: &[C],
    build: impl Fn(C) -> Candidate,
) -> (Vec<Compiled>, Vec<Option<u64>>) {
    let store = TuneStore::global();
    let mut bench: Bench<C> = Bench::default();
    let verdicts: Vec<Compiled> = ranked.chunks(64).flat_map(|chunk| bench.admit(chunk, &build, store)).collect();
    let mut runs: Vec<C> =
        ranked.iter().zip(&verdicts).filter(|(_, v)| **v != Compiled::Failed).map(|(c, _)| *c).collect();
    runs.extend(extra.iter().filter(|c| !runs.contains(c)).copied().collect::<Vec<_>>());
    // What each run's programs emit, where the compiles learned it, an
    // index of its own where not: twins time once, the rest apart.
    let keys: Vec<u64> =
        runs.iter().enumerate().map(|(i, &c)| store.emitted_key(&build(c)).unwrap_or(i as u64 + 1)).collect();
    let passes: Vec<Vec<Option<u64>>> = (0..2)
        .map(|_| bench.time(&runs, runs.iter().map(|&c| build(c)).collect(), &keys, Pass::Whole, store))
        .collect();
    let ns = |c: &C| -> Option<u64> {
        let at = runs.iter().position(|r| r == c)?;
        passes.iter().filter_map(|p| p[at]).min()
    };
    (verdicts, ranked.iter().chain(extra).map(ns).collect())
}

fn report<C: std::fmt::Debug>(
    label: &str,
    ranked: &[C],
    predicted: &[f64],
    verdicts: &[Compiled],
    ns: &[Option<u64>],
    pick: &C,
) {
    let show = |t: &Option<u64>| t.map_or("-".to_string(), |t| t.to_string());
    for (rank, c) in ranked.iter().enumerate() {
        eprintln!("fit|{label}|{rank}|{:?}|{:.1}|{}|{c:?}", verdicts[rank], predicted[rank], show(&ns[rank]));
    }
    eprintln!("pick|{label}|{}|{pick:?}", show(&ns[ranked.len()]));
}

/// The GEMM shapes: `(label, m, n, k, epilogue)`.
fn gemm_shapes() -> Vec<(&'static str, usize, usize, usize, Epilogue)> {
    let plain = Epilogue::default();
    let gelu = Epilogue { act: Act::Gelu, ..plain };
    let residual = Epilogue { residual: true, ..plain };
    let swiglu = Epilogue { act: Act::Silu, gated: true, ..plain };
    vec![
        ("gemm4096", 4096, 4096, 4096, plain),
        ("nemotron_qkv", 704, 1536, 512, plain),
        ("nemotron_ff1", 704, 2048, 512, gelu),
        ("nemotron_ff2", 704, 512, 2048, residual),
        ("qwen3_qkv", 3072, 4096, 1024, plain),
        ("qwen3_gate_up", 3072, 3072, 1024, swiglu),
        ("qwen3_down", 3072, 1024, 3072, residual),
        ("whisper_fc1", 1500, 5120, 1280, gelu),
        ("whisper_fc2", 1500, 1280, 5120, residual),
    ]
}

/// The attention shapes: `(label, batch, heads, t, d, causal)`.
const ATTENTION_SHAPES: [(&str, usize, usize, usize, usize, bool); 6] = [
    ("fa_d64", 4, 8, 2048, 64, false),
    ("fa_d64_causal", 4, 8, 2048, 64, true),
    ("fa_d128", 4, 8, 2048, 128, false),
    ("fa_d128_causal", 4, 8, 2048, 128, true),
    ("whisper_enc_attn", 1, 20, 1500, 64, false),
    ("qwen3_attn", 8, 16, 384, 128, true),
];

/// Whether `COST_MODEL_ONLY` (comma-separated labels) selects `label`; all
/// do without it.
fn wanted(label: &str) -> bool {
    std::env::var("COST_MODEL_ONLY").ok().is_none_or(|o| o.split(',').any(|w| w == label))
}

/// A shape's ranking: its label, its configs in ranked order, and the ns
/// the model ranks each at.
type Ranking<C> = (&'static str, Vec<C>, Vec<f64>);

/// Every shape's ranking under `planner`, GEMMs then attention.
fn rankings(planner: &Planner) -> (Vec<Ranking<GemmCfg>>, Vec<Ranking<FaCfg>>) {
    let gemms = gemm_shapes()
        .into_iter()
        .filter(|s| wanted(s.0))
        .map(|(label, m, n, k, epilogue)| {
            let p = GemmProblem { batches: 1, m, n, k, halves: if epilogue.gated { 2 } else { 1 }, rope: None };
            let ranked: Vec<GemmCfg> = planner.gemm_ranked(1, m, n, k, epilogue.gated).to_vec();
            let ns = ranked.iter().map(|c| p.admit(planner, c).expect("ranked").0).collect();
            (label, ranked, ns)
        })
        .collect();
    let attention = ATTENTION_SHAPES
        .into_iter()
        .filter(|s| wanted(s.0))
        .map(|(label, batch, heads, t, d, causal)| {
            let p = AttentionProblem { tiles: batch * heads, t, tk: t, d, causal, bias: false };
            let ranked: Vec<FaCfg> = planner.attention_ranked(batch * heads, t, t, d, causal, false).to_vec();
            let ns = ranked.iter().map(|c| p.admit(planner, c).expect("ranked").0).collect();
            (label, ranked, ns)
        })
        .collect();
    (gemms, attention)
}

/// The configs [`cost_model_probe`] times beside the tuner's pick: the
/// whole ranking, or none under `COST_MODEL_PICKS` (an A/B of the picks
/// across builds, whose rankings the replay prices).
fn timing<C>(ranked: &[C]) -> &[C] {
    if std::env::var_os("COST_MODEL_PICKS").is_some() { &[] } else { ranked }
}

#[test]
#[ignore = "perf probe: needs a GPU"]
fn cost_model_probe() {
    let Some(planner) = super::device_planner() else { return };
    eprintln!("costs|{:?}|unmeasured {:?}", planner.device.costs, planner.unmeasured);
    let (gemms, attentions) = rankings(&planner);
    for ((label, m, n, k, epilogue), (_, ranked, predicted)) in
        gemm_shapes().into_iter().filter(|s| wanted(s.0)).zip(gemms)
    {
        let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue, cfg: super::device::BIG };
        let pick = super::device::device_gemms(&planner, &spec).1;
        let build = |cfg| vec![(gemm::<BF16>(&GemmSpec { cfg, ..spec.clone() }), cfg.lowering(&planner.device))];
        let ranked = timing(&ranked);
        let (verdicts, ns) = timed(ranked, &[pick], build);
        report(label, ranked, &predicted, &verdicts, &ns, &pick);
    }
    for ((label, batch, heads, t, d, causal), (_, ranked, predicted)) in
        ATTENTION_SHAPES.into_iter().filter(|s| wanted(s.0)).zip(attentions)
    {
        let mask = AttnMask { causal, ..AttnMask::default() };
        let spec = super::attention::probe_spec(batch, heads, t, t, d, mask);
        let pick = super::attention::device_attention(&planner, &spec).2;
        let build = |cfg| vec![(attention::<BF16>(&AttnSpec { cfg, ..spec.clone() }), cfg.lowering(&planner.device))];
        let ranked = timing(&ranked);
        let (verdicts, ns) = timed(ranked, &[pick], build);
        report(label, ranked, &predicted, &verdicts, &ns, &pick);
    }
}

/// [`cost_model_probe`]'s rankings on the host for a card (`COST_MODEL_CARD`,
/// a name of [`crate::atoms::CARDS`]) under the costs its device stored
/// (`COST_MODEL_STORE`, a tune store directory): one `fit|label|rank|-|ns|-|cfg`
/// line a config, which a probe's log of the same card times; a model
/// changed since prices the configs it timed without the device.
#[test]
#[ignore = "replays a card's store: needs COST_MODEL_CARD and COST_MODEL_STORE"]
fn cost_model_replay() {
    let card = std::env::var("COST_MODEL_CARD").expect("COST_MODEL_CARD");
    let store = std::env::var("COST_MODEL_STORE").expect("COST_MODEL_STORE");
    let target = crate::atoms::Target::card(&card).expect("a listed card");
    let measured = crate::probe::stored(std::path::Path::new(&store), &target).expect("the card's store file");
    let planner = Planner::measured(target, measured).expect("an SM count");
    eprintln!("costs|{:?}|unmeasured {:?}", planner.device.costs, planner.unmeasured);
    let (gemms, attentions) = rankings(&planner);
    let target = &planner.device.target;
    // The blocks an SM runs at once: those it holds, at most its share.
    let at_once = |blocks: usize, smem: usize, regs: u32, waves: u32| {
        planner.resident(smem, regs, waves as usize).max(1).min(blocks.div_ceil(planner.sms()))
    };
    let print = |label: &str, ranked: Vec<(String, usize)>, ns: Vec<f64>| {
        for (rank, ((c, at_once), ns)) in ranked.iter().zip(ns).enumerate() {
            eprintln!("fit|{label}|{rank}|-|{ns:.1}|-|{c}|{at_once}");
        }
    };
    for ((label, ranked, ns), (_, m, n, _, epilogue)) in
        gemms.into_iter().zip(gemm_shapes().into_iter().filter(|s| wanted(s.0)))
    {
        let halves = if epilogue.gated { 2 } else { 1 };
        let row = |c: &GemmCfg| {
            let blocks = m.div_ceil(c.tile[0]) * n.div_ceil(c.tile[1]);
            let regs = c.registers(target, halves).expect("ranked");
            (format!("{c:?}"), at_once(blocks, c.smem_bytes(epilogue.gated, 2), regs, c.warps[0] * c.warps[1]))
        };
        print(label, ranked.iter().map(row).collect(), ns);
    }
    for ((label, ranked, ns), (_, batch, heads, t, d, _)) in
        attentions.into_iter().zip(ATTENTION_SHAPES.into_iter().filter(|s| wanted(s.0)))
    {
        let row = |c: &FaCfg| {
            let (smem, regs) = (c.smem_bytes(d, false), c.registers(target, d, false).expect("ranked"));
            (format!("{c:?}"), at_once(batch * heads * t.div_ceil(c.bq), smem, regs, c.warps()))
        };
        print(label, ranked.iter().map(row).collect(), ns);
    }
}

/// The trip pairs at 1..48 blocks an SM, each probe's SM time per
/// block-trip (`trip`) and the wall time a round of them adds per trip
/// (`wall`, the SM time times the blocks): flat SM time is per-block
/// issue, SM time falling as the blocks grow is a wait the resident blocks
/// hide. Prints, never asserts.
#[test]
#[ignore = "perf probe: needs a GPU"]
fn trip_by_residency_probe() {
    let Some(target) = super::device_target() else { return };
    for (op, unroll, stages) in [("probe:trip", true, 2), ("probe:trip_rolled", false, 2), ("probe:trip_one", false, 1)]
    {
        for per_sm in [1usize, 2, 3, 4, 6, 8, 12, 24, 48] {
            let p = crate::probe::trips_at(&target, op, unroll, stages, per_sm).expect("an SM count");
            let mut built = crate::tune::compile(p.candidates.clone(), &crate::tune::ScratchPool::default());
            let pairs = crate::probe::resolve_pairs(&mut built);
            let Some(r) = crate::probe::resolved(&pairs) else {
                eprintln!("residency|{op}|{per_sm}|unresolved {pairs:?}");
                continue;
            };
            let unit = p.scale / p.units;
            eprintln!(
                "residency|{op}|{per_sm}|trip {:.3}|wall {:.3}|spread {:.3}|base {:.3}|measurable {}",
                r.difference * unit,
                r.difference * unit * per_sm as f64,
                r.spread * unit,
                r.baseline * unit,
                r.measurable(),
            );
        }
    }
}

/// The key blocks the kernel runs for a causal query block at `q_off`: up to
/// the block holding its last query's key, at most all of them (the mask is
/// aligned at the first query and key, `kernels::attention`'s `end`).
fn kernel_trips(t: usize, tk: usize, bq: usize, bkv: usize) -> f64 {
    let blocks: Vec<usize> = (0..t.div_ceil(bq)).map(|j| tk.div_ceil(bkv).min((j * bq + bq).div_ceil(bkv))).collect();
    blocks.iter().sum::<usize>() as f64 / blocks.len() as f64
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// A causal block's trips as the model counts them hold the kernel's
    /// mean over the query blocks (an appended cache, `t < tk`, included:
    /// its queries see the first keys, not the last), and its diagonal
    /// stays within the [`DIAGONAL`] trips a key block of it is charged.
    #[test]
    fn causal_trips_hold_the_kernels(
        t in 1usize..3000,
        tk in 1usize..5000,
        bq in prop::sample::select(vec![16usize, 32, 64, 128]),
        bkv in prop::sample::select(vec![16usize, 32, 64, 128]),
    ) {
        let p = AttentionProblem { tiles: 1, t, tk, d: 64, causal: true, bias: false };
        let model = p.causal_trips(bq, bkv) as f64;
        let diagonal = (DIAGONAL * bq.div_ceil(bkv)) as f64;
        let kernel = kernel_trips(t, tk, bq, bkv);
        prop_assert!(kernel <= model + 1.0, "kernel {kernel}, model {model}");
        prop_assert!(model - diagonal <= kernel + 1.0, "kernel {kernel}, model {model}");
    }
}

/// Where queries and keys are one sequence the count is the one before
/// the mask's alignment was counted: half the key blocks and the diagonal.
#[test_case::test_case(2048, 64, 64; "d64 blocks")]
#[test_case::test_case(1500, 128, 32; "whisper")]
#[test_case::test_case(384, 16, 128; "qwen3")]
fn causal_trips_of_one_sequence_are_half_the_keys(t: usize, bq: usize, bkv: usize) {
    let p = AttentionProblem { tiles: 1, t, tk: t, d: 64, causal: true, bias: false };
    assert_eq!(p.causal_trips(bq, bkv), t.div_ceil(2 * bkv) + DIAGONAL * bq.div_ceil(bkv));
}

/// Pairs of GEMM configs the model orders as the device measured them
/// under that device's costs, the faster first (`cost_model_probe` on three
/// fresh stores each, 2026-10-19), each a block alone on its SM against a
/// tile the SM runs several of at once: a lone block's trip waits out what
/// other blocks' work hides ([`Costs::trip_at`]), and a lone one-slot
/// `cp.async` block fills after its products. Before either term the model
/// ranked every slow one first.
#[test_case::test_case(crate::test::unit::sm86_costs(), "rtx3060", [3072, 3072, 1024, 2],
    gemm_cfg([64, 64, 32], 2, [4, 1], true), gemm_cfg([128, 64, 64], 2, [4, 2], false);
    "sm86 Qwen3 gate-up: 1.52 ms at three an SM, 2.01 alone")]
#[test_case::test_case(crate::test::unit::sm86_costs(), "rtx3060", [3072, 3072, 1024, 2],
    gemm_cfg([64, 64, 64], 1, [4, 1], false), gemm_cfg([64, 128, 64], 1, [4, 4], false);
    "sm86 Qwen3 gate-up one slot: 1.55 ms at two an SM, 2.23 alone")]
#[test_case::test_case(crate::test::unit::gfx1201_costs(), "rx9070xt", [1500, 1280, 5120, 1],
    gemm_cfg([128, 128, 32], 2, [4, 2], false), gemm_cfg([256, 128, 32], 2, [2, 4], false);
    "gfx1201 Whisper fc2: 156 us at two a CU, 172 alone")]
fn a_block_alone_ranks_behind_the_measured_faster_tile(
    costs: Costs,
    card: &str,
    [m, n, k, halves]: [usize; 4],
    fast: GemmCfg,
    slow: GemmCfg,
) {
    let target = Target::card(card).expect("a listed card");
    let planner = Planner::new(Device { target, costs }).expect("an SM count");
    let p = GemmProblem { batches: 1, m, n, k, halves, rope: None };
    let cost = |c: &GemmCfg| p.admit(&planner, c).expect("admitted").0;
    assert!(cost(&fast) < cost(&slow), "{} against {}", cost(&fast), cost(&slow));
}

/// The side stream's contract: bytes of none price at nothing (a zero
/// stream leaves the run's price and floor exactly what they were, whatever
/// share it claims), side bytes price above none at the ladder of the
/// working set they name (a stream cache-resident at its working set
/// cheaper than one that spills past the ladder's last rung), and the
/// stream's distinct bytes are bytes the kernel moves: they count toward
/// the run's floor.
#[test]
fn a_side_stream_prices_alongside_the_epilogue() {
    use crate::ops::config::Stream;
    let target = Target::card("rtx3060").expect("a listed card");
    let planner = Planner::new(Device { target, costs: crate::test::unit::sm86_costs() }).expect("an SM count");
    let p = GemmProblem { batches: 1, m: 4096, n: 4096, k: 1024, halves: 1 };
    let cfg = planner.gemm_ranked(1, 4096, 4096, 1024, false)[0];
    let mut claim = p.claim(&planner, &cfg).expect("ranked");
    let (plain, floor) = (planner.price(&claim), planner.floor(&claim));
    assert_eq!((claim.side.bytes, claim.side.unique), (0.0, 0.0));
    claim.side = Stream { bytes: 0.0, unique: 0.0, shared: 1.0 };
    assert_eq!((planner.price(&claim), planner.floor(&claim)), (plain, floor));
    claim.side = Stream { bytes: (1 << 20) as f64, unique: (1 << 20) as f64, shared: 0.0 };
    let resident = (planner.price(&claim), planner.floor(&claim));
    assert!(resident.0 > plain, "{} against {plain}", resident.0);
    assert!(resident.1 > floor, "{} against {floor}", resident.1);
    claim.side.unique = 1e12;
    let spilled = planner.price(&claim);
    assert!(spilled > resident.0, "a working set past the ladder priced dearer: {spilled} against {}", resident.0);
}
