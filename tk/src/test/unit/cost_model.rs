//! The planner's cost model on the host: its terms from fixture costs, and
//! the pairs devices measured (`cargo run --example tk_bench -- fit`) ranked
//! as the devices ran them.

use proptest::prelude::*;

use crate::atoms::{Costs, Device, Target};
use crate::kernels::Act;
use crate::kernels::gemm::{Epilogue, GemmCfg};
use crate::ops::config::{AttentionProblem, DIAGONAL, GemmProblem, Planner, gemm_cfg};

/// The probed layers' epilogues beside the plain one.
const RESIDUAL: Epilogue = Epilogue { residual: true, ..Epilogue::DEFAULT };
const SWIGLU: Epilogue = Epilogue { act: Act::Silu, gated: true, ..Epilogue::DEFAULT };

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
/// under that device's costs, the faster first (`tk_bench fit` on three
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
