//! The planner's cost model: its terms on the host, from fixture costs and
//! the pairs devices measured; and against the device, every config a
//! ranking admits for a few GEMM and attention shapes (4096³, model
//! projections, flash attention at d64/d128 with and without a causal mask)
//! compiled and timed beside the ns the model ranks it at, and the tuner's
//! pick timed with them: one `fit|…` line a config and a `pick|…` line a
//! shape, for an offline fit of rank correlation and regret (printed only).

use proptest::prelude::*;

use crate::atoms::{Costs, Device, Target};
use crate::build::BF16;
use crate::kernels::attention::{AttnMask, FaCfg, attention};
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::kernels::{Act, Batch};
use crate::ops::config::{AttentionProblem, DIAGONAL, GemmProblem, Planner, gemm_cfg};
use crate::sweep::assumed;
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

/// The probed layers' epilogues beside the plain one.
const RESIDUAL: Epilogue = Epilogue { residual: true, ..Epilogue::DEFAULT };
const SWIGLU: Epilogue = Epilogue { act: Act::Silu, gated: true, ..Epilogue::DEFAULT };

/// The GEMM shapes: `(label, m, n, k, epilogue)`.
fn gemm_shapes() -> Vec<(&'static str, usize, usize, usize, Epilogue)> {
    let plain = Epilogue::default();
    let gelu = Epilogue { act: Act::Gelu, ..plain };
    let (residual, swiglu) = (RESIDUAL, SWIGLU);
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
            let p = GemmProblem::new(1, m, n, k, &epilogue);
            let ranked: Vec<GemmCfg> = planner.gemm_ranked(1, m, n, k, &epilogue).to_vec();
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
        let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue };
        let pick = super::device::device_gemms(&planner, &spec).1;
        let build = |cfg| vec![(gemm::<BF16>(&assumed(spec.clone(), cfg)), cfg.lowering(&planner.device))];
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
        let build = |cfg| vec![(attention::<BF16>(&assumed(spec.clone(), cfg)), cfg.lowering(&planner.device))];
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

/// Where queries and keys are one sequence the count is half the key
/// blocks and the diagonal.
#[test_case::test_case(2048, 64, 64; "d64 blocks")]
#[test_case::test_case(1500, 128, 32; "whisper")]
#[test_case::test_case(384, 16, 128; "qwen3")]
fn causal_trips_of_one_sequence_are_half_the_keys(t: usize, bq: usize, bkv: usize) {
    let p = AttentionProblem { tiles: 1, t, tk: t, d: 64, causal: true, bias: false };
    assert_eq!(p.causal_trips(bq, bkv), t.div_ceil(2 * bkv) + DIAGONAL * bq.div_ceil(bkv));
}

/// Pairs of GEMM configs the model orders as the device measured them
/// under that device's costs, the faster first (`cost_model_probe` on three
/// fresh stores each), each a block alone on its SM against a
/// tile the SM runs several of at once, under the layer's epilogue the
/// probe ran: a lone block's trip waits out what other blocks' work hides
/// ([`Costs::trip_at`]), a lone one-slot `cp.async` block fills after its
/// products, and resident blocks hide one another's wait for their loads
/// ([`Costs::wait_hiding`]).
#[test_case::test_case(crate::test::unit::sm86_costs(), "rtx3060", [3072, 3072, 1024], SWIGLU,
    gemm_cfg([64, 64, 32], 2, [4, 1], true), gemm_cfg([128, 64, 64], 2, [4, 2], false);
    "sm86 Qwen3 gate-up: 1.52 ms at three an SM, 2.01 alone")]
#[test_case::test_case(crate::test::unit::sm86_costs(), "rtx3060", [3072, 3072, 1024], SWIGLU,
    gemm_cfg([64, 64, 64], 1, [4, 1], false), gemm_cfg([64, 128, 64], 1, [4, 4], false);
    "sm86 Qwen3 gate-up one slot: 1.55 ms at two an SM, 2.23 alone")]
#[test_case::test_case(crate::test::unit::gfx1201_costs(), "rx9070xt", [1500, 1280, 5120], RESIDUAL,
    gemm_cfg([128, 128, 32], 2, [4, 2], false), gemm_cfg([256, 128, 32], 2, [2, 4], false);
    "gfx1201 Whisper fc2: 156 us at two a CU, 172 alone")]
fn a_block_alone_ranks_behind_the_measured_faster_tile(
    costs: Costs,
    card: &str,
    [m, n, k]: [usize; 3],
    epilogue: Epilogue,
    fast: GemmCfg,
    slow: GemmCfg,
) {
    let target = Target::card(card).expect("a listed card");
    let planner = Planner::new(Device { target, costs }).expect("an SM count");
    let p = GemmProblem::new(1, m, n, k, &epilogue);
    let cost = |c: &GemmCfg| p.admit(&planner, c).expect("admitted").0;
    assert!(cost(&fast) < cost(&slow), "{} against {}", cost(&fast), cost(&slow));
}

/// The residency term alone, at equal work: one claim priced a block alone
/// on its SM and several at once (its shared memory alone setting how many
/// an SM holds), less the same claim at no memory latency, leaves the run's
/// waits — a round's wait a round alone, and at `n` blocks at once the
/// rounds' waits times `n` to the power of one less the card's
/// [`Costs::wait_hiding`], the power its trip probes measured a full SM
/// hiding a lone block's trip by. Where that power is one (gfx1201, whose
/// probes hid less; a lone trip unmeasured) the rounds wait once each.
#[test_case::test_case(crate::atoms::sm86(), crate::test::unit::sm86_costs(), 2, 1.710; "sm86 two slots")]
#[test_case::test_case(crate::atoms::sm86(), crate::test::unit::sm86_costs(), 1, 1.489; "sm86 one cp.async slot")]
#[test_case::test_case(Target::card("mi300x").unwrap(), crate::test::unit::gfx942_costs(), 1, 1.118;
    "gfx942 one register-staged slot, hidden as two")]
#[test_case::test_case(Target::card("rx9070xt").unwrap(), crate::test::unit::gfx1201_costs(), 2, 1.0;
    "gfx1201 at a round's wait")]
#[test_case::test_case(crate::atoms::sm86(), Costs { trip_lone: 0.0, ..crate::test::unit::sm86_costs() }, 2, 1.0;
    "a lone trip unmeasured")]
fn resident_blocks_hide_one_anothers_waits(target: Target, costs: Costs, stages: usize, power: f64) {
    let serial = stages <= 1 && target.fill == crate::atoms::Fill::CpAsync;
    let hiding = costs.wait_hiding(serial, crate::probe::trip_blocks(&target));
    assert!((hiding - power).abs() < 1e-3, "hiding {hiding}, expected {power}");
    let planner = Planner::new(Device { target: target.clone(), costs }).expect("an SM count");
    let still = Costs { latency: 0.0, latency_one: 0.0, ..costs };
    let still = Planner::new(Device { target: target.clone(), costs: still }).expect("an SM count");
    let p = GemmProblem::new(1, 4096, 4096, 1024, &Epilogue::DEFAULT);
    let claim = planner.claim_of(&p, &gemm_cfg([64, 64, 32], stages, [2, 2], false)).expect("admitted");
    let per_sm = claim.blocks.div_ceil(planner.sms());
    // The run's waits of `claim` with `smem` a block, and its blocks at once.
    let waits = |smem: usize| {
        let c = crate::ops::config::Claim { smem, ..claim };
        let at_once = planner.resident(smem, c.regs, c.waves).max(1).min(per_sm);
        (planner.price(&c) - still.price(&c), at_once)
    };
    let (alone, one) = waits(target.smem_bytes);
    let (resident, n) = waits(target.smem_bytes / 3);
    assert_eq!(one, 1);
    assert!(n >= 2, "{n} blocks at once");
    // Alone, a block waits its round's whole wait every trip, whatever the card hides.
    let wait = if serial { costs.latency_one.max(costs.latency) } else { costs.latency / (stages.max(2) - 1) as f64 };
    let lone = (claim.trips * per_sm) as f64 * wait;
    assert!((alone / lone - 1.0).abs() < 1e-9, "{alone} against {lone}");
    let rounds = per_sm.div_ceil(n) as f64 / per_sm as f64;
    let want = rounds * (n as f64).powf(1.0 - hiding);
    assert!((resident / alone / want - 1.0).abs() < 1e-9, "{resident} / {alone} against {want}");
    assert!(alone > resident, "{alone} against {resident}");
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
    let p = GemmProblem::new(1, 4096, 4096, 1024, &Epilogue::DEFAULT);
    let cfg = planner.gemm_ranked(1, 4096, 4096, 1024, &Epilogue::DEFAULT)[0];
    let mut claim = planner.claim_of(&p, &cfg).expect("ranked");
    let (plain, floor) = (planner.price(&claim), planner.floor(&claim));
    assert_eq!((claim.side.bytes, claim.side.unique), (0.0, 0.0));
    claim.side = Stream { bytes: 0.0, unique: 0.0, shared: 1.0, ..Stream::default() };
    assert_eq!((planner.price(&claim), planner.floor(&claim)), (plain, floor));
    claim.side = Stream { bytes: (1 << 20) as f64, unique: (1 << 20) as f64, shared: 0.0, ..Stream::default() };
    let resident = (planner.price(&claim), planner.floor(&claim));
    assert!(resident.0 > plain, "{} against {plain}", resident.0);
    assert!(resident.1 > floor, "{} against {floor}", resident.1);
    claim.side.unique = 1e12;
    let spilled = planner.price(&claim);
    assert!(spilled > resident.0, "a working set past the ladder priced dearer: {spilled} against {}", resident.0);
}

/// An SM holds no more blocks than its hardware takes whatever their
/// resources: small blocks of one or two warps fill sm_86's wave slots
/// (48) past its sixteen blocks and stop there, sm_120's at 32, while four
/// warps stop at the wave slots first; an AMD block of one wave takes no
/// barrier and fills the CU's wave slots, of two waves the slots and the
/// sixteen barriers alike.
#[test_case::test_case("rtx3060", 1 => 16; "sm_86 one warp")]
#[test_case::test_case("rtx3060", 2 => 16; "sm_86 two warps")]
#[test_case::test_case("rtx3060", 4 => 12; "sm_86 four warps")]
#[test_case::test_case("rtx5090", 1 => 32; "sm_120 one warp")]
#[test_case::test_case("rx9070xt", 1 => 32; "gfx1201 one wave")]
#[test_case::test_case("rx9070xt", 2 => 16; "gfx1201 two waves")]
#[test_case::test_case("mi300x", 1 => 32; "gfx942 one wave")]
fn an_sm_holds_its_hardwares_blocks_at_most(card: &str, waves: usize) -> usize {
    let target = Target::card(card).expect("a listed card");
    let planner = Planner::new(Device { target, costs: crate::test::unit::costs() }).expect("an SM count");
    planner.resident(16, 16, waves)
}
