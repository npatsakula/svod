//! The planner's cost model against the device: every config a ranking
//! admits for a few GEMM and attention shapes (4096³, model projections,
//! flash attention at d64/d128 with and without a causal mask) compiled and
//! timed beside the ns the model ranks it at, and the tuner's pick timed
//! with them. One `fit|…` line a config and a `pick|…` line a shape, for an
//! offline fit of rank correlation and regret; never asserts. And the
//! model's terms on the host, from fixture costs.

use proptest::prelude::*;

use crate::build::BF16;
use crate::kernels::attention::{AttnMask, AttnSpec, FaCfg, attention};
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::kernels::{Act, Batch};
use crate::ops::config::{Attention, DIAGONAL, Planner, Problem};
use crate::tune::{Bench, Candidate, Compiled};

/// Each of `ranked` compiled (in ranked batches, as the tuner admits), then
/// every one that compiled timed twice, its lesser time kept: the verdicts
/// and ns in `ranked`'s order, and the extra candidates' ns after them.
fn timed<C: Copy + PartialEq + std::fmt::Debug>(
    ranked: &[C],
    extra: &[C],
    build: impl Fn(C) -> Candidate,
) -> (Vec<Compiled>, Vec<Option<u64>>) {
    let mut bench: Bench<C> = Bench::default();
    let verdicts: Vec<Compiled> = ranked.chunks(64).flat_map(|chunk| bench.admit(chunk, &build)).collect();
    let mut runs: Vec<C> =
        ranked.iter().zip(&verdicts).filter(|(_, v)| **v != Compiled::Failed).map(|(c, _)| *c).collect();
    runs.extend(extra.iter().filter(|c| !runs.contains(c)).copied().collect::<Vec<_>>());
    let passes: Vec<Vec<Option<u64>>> =
        (0..2).map(|_| bench.time(&runs, runs.iter().map(|&c| build(c)).collect())).collect();
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
            let p = Problem { batches: 1, m, n, k, halves: if epilogue.gated { 2 } else { 1 } };
            let ranked: Vec<GemmCfg> = planner.gemm_ranked(1, m, n, k, epilogue.gated).to_vec();
            let ns = ranked.iter().map(|c| p.admit(planner, c).expect("ranked").0).collect();
            (label, ranked, ns)
        })
        .collect();
    let attention = ATTENTION_SHAPES
        .into_iter()
        .filter(|s| wanted(s.0))
        .map(|(label, batch, heads, t, d, causal)| {
            let p = Attention { tiles: batch * heads, t, tk: t, d, causal, bias: false };
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
    let print = |label: &str, ranked: Vec<String>, ns: Vec<f64>| {
        for (rank, (c, ns)) in ranked.iter().zip(ns).enumerate() {
            eprintln!("fit|{label}|{rank}|-|{ns:.1}|-|{c}");
        }
    };
    for (label, ranked, ns) in gemms {
        print(label, ranked.iter().map(|c| format!("{c:?}")).collect(), ns);
    }
    for (label, ranked, ns) in attentions {
        print(label, ranked.iter().map(|c| format!("{c:?}")).collect(), ns);
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
        let p = Attention { tiles: 1, t, tk, d: 64, causal: true, bias: false };
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
    let p = Attention { tiles: 1, t, tk: t, d: 64, causal: true, bias: false };
    assert_eq!(p.causal_trips(bq, bkv), t.div_ceil(2 * bkv) + DIAGONAL * bq.div_ceil(bkv));
}
