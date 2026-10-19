//! The planner's cost model against the device: every config a ranking
//! admits for a few GEMM and attention shapes (4096³, model projections,
//! flash attention at d64/d128 with and without a causal mask) compiled and
//! timed beside the ns the model ranks it at, and the tuner's pick timed
//! with them. One `fit|…` line a config and a `pick|…` line a shape, for an
//! offline fit of rank correlation and regret; never asserts.

use crate::build::BF16;
use crate::kernels::attention::{AttnMask, AttnSpec, FaCfg, attention};
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::kernels::{Act, Batch};
use crate::ops::config::{Attention, Problem};
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

#[test]
#[ignore = "perf probe: needs a GPU"]
fn cost_model_probe() {
    let Some(planner) = super::device_planner() else { return };
    eprintln!("costs|{:?}|unmeasured {:?}", planner.device.costs, planner.unmeasured);
    let plain = Epilogue::default();
    let gelu = Epilogue { act: Act::Gelu, ..plain };
    let residual = Epilogue { residual: true, ..plain };
    let swiglu = Epilogue { act: Act::Silu, gated: true, ..plain };
    let only = std::env::var("COST_MODEL_ONLY").ok();
    let wanted = |label: &str| only.as_deref().is_none_or(|o| o.split(',').any(|w| w == label));
    for (label, m, n, k, epilogue) in [
        ("gemm4096", 4096, 4096, 4096, plain),
        ("nemotron_qkv", 704, 1536, 512, plain),
        ("nemotron_ff1", 704, 2048, 512, gelu),
        ("nemotron_ff2", 704, 512, 2048, residual),
        ("qwen3_qkv", 3072, 4096, 1024, plain),
        ("qwen3_gate_up", 3072, 3072, 1024, swiglu),
        ("qwen3_down", 3072, 1024, 3072, residual),
        ("whisper_fc1", 1500, 5120, 1280, gelu),
        ("whisper_fc2", 1500, 1280, 5120, residual),
    ] {
        if !wanted(label) {
            continue;
        }
        let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue, cfg: super::device::BIG };
        let p = Problem { batches: 1, m, n, k, halves: if epilogue.gated { 2 } else { 1 } };
        let ranked: Vec<GemmCfg> = planner.gemm_ranked(1, m, n, k, epilogue.gated).to_vec();
        let predicted: Vec<f64> = ranked.iter().map(|c| p.admit(&planner, c).expect("ranked").0).collect();
        let pick = super::device::device_gemms(&planner, &spec).1;
        let build = |cfg| vec![(gemm::<BF16>(&GemmSpec { cfg, ..spec.clone() }), cfg.lowering(&planner.device))];
        let (verdicts, ns) = timed(&ranked, &[pick], build);
        report(label, &ranked, &predicted, &verdicts, &ns, &pick);
    }
    for (label, batch, heads, t, d, causal) in [
        ("fa_d64", 4, 8, 2048, 64, false),
        ("fa_d64_causal", 4, 8, 2048, 64, true),
        ("fa_d128", 4, 8, 2048, 128, false),
        ("fa_d128_causal", 4, 8, 2048, 128, true),
        ("whisper_enc_attn", 1, 20, 1500, 64, false),
        ("qwen3_attn", 8, 16, 384, 128, true),
    ] {
        if !wanted(label) {
            continue;
        }
        let mask = AttnMask { causal, ..AttnMask::default() };
        let spec = super::attention::probe_spec(batch, heads, t, t, d, mask);
        let p = Attention { tiles: batch * heads, t, tk: t, d, causal, bias: false };
        let ranked: Vec<FaCfg> = planner.attention_ranked(batch * heads, t, t, d, causal, false).to_vec();
        let predicted: Vec<f64> = ranked.iter().map(|c| p.admit(&planner, c).expect("ranked").0).collect();
        let pick = super::attention::device_attention(&planner, &spec).2;
        let build = |cfg| vec![(attention::<BF16>(&AttnSpec { cfg, ..spec.clone() }), cfg.lowering(&planner.device))];
        let (verdicts, ns) = timed(&ranked, &[pick], build);
        report(label, &ranked, &predicted, &verdicts, &ns, &pick);
    }
}
