use proptest::prelude::*;
use svod_dtype::ScalarDType;
use test_case::test_case;

use crate::atoms::{Core, Fill, SyncCaps};
use crate::build::BF16;
use crate::interp::{round_to, run};
use crate::ir::*;
use crate::kernels::attention::{AttnMask, AttnSpec, Bias, FaCfg, attention};
use crate::kernels::gemm::{GemmCfg, gemm};
use crate::schedule::expand;
use crate::sweep::assumed;

fn count<P: Phase>(prog: &Program<P>, pred: impl Fn(&Stmt<P>) -> bool) -> usize {
    prog.walk().filter(|(_, s)| pred(s)).count()
}

/// A GEMM's bf16 `a [m, k]` and `b [n, k]` and its zeroed `c [m, n]`.
fn operands(m: usize, n: usize, k: usize) -> Vec<Vec<f64>> {
    let a = (0..m * k).map(|i| round_to(ScalarDType::BFloat16, ((i * 13) % 17) as f64 / 8.0 - 1.0)).collect();
    let b = (0..n * k).map(|i| round_to(ScalarDType::BFloat16, ((i * 7) % 19) as f64 / 8.0 - 1.0)).collect();
    vec![a, b, vec![0.0; m * n]]
}

/// Expanding the pipeline keeps the GEMM's result (the interpreter treats
/// synchronization as no-ops, so this checks the step/slot bookkeeping and
/// the prologue and drain, guarded or peeled) and leaves no pipeline behind.
#[test_case(Fill::CpAsync, 1; "cp.async one stage")]
#[test_case(Fill::CpAsync, 2; "cp.async two stages")]
#[test_case(Fill::CpAsync, 4; "cp.async four stages")]
#[test_case(Fill::Staged, 2; "register staged")]
#[test_case(Fill::Staged, 1; "register staged, one slot")]
fn expansion_preserves_the_result(fill: Fill, stages: usize) {
    let (m, n, k) = (64usize, 32usize, 160usize);
    let prog = super::gemm_nt(m, n, k, 32, 32, 32, stages);
    let want = run(&prog, operands(m, n, k), &[("b", 1)]).unwrap();
    let prog = expand(prog, fill, Core::Warp, SyncCaps::default());
    let got = run(&prog, operands(m, n, k), &[("b", 1)]).unwrap();
    assert_eq!(got[2], want[2]);
    assert_eq!(count(&prog, |s| matches!(s, Stmt::Pipeline(_))), 0);
    let staged = fill == Fill::Staged;
    assert_eq!(count(&prog, |s| matches!(s, Stmt::Loop(_))), if staged { 3 } else { 1 }, "peeled or one loop");
    let barriers = count(&prog, |s| matches!(s, Stmt::Sync(Sync::Barrier { .. })));
    let want = if staged && stages == 1 {
        3
    } else if staged || (fill == Fill::CpAsync && stages == 1) {
        2
    } else {
        1
    };
    assert_eq!(barriers, want, "one per trip (two around a one-slot commit), plus the peeled prologue's");
    if fill == Fill::CpAsync && stages >= 2 {
        let waits: Vec<_> = prog
            .walk()
            .filter_map(|(_, s)| match s {
                Stmt::Sync(Sync::WaitAsync { pending }) => Some(*pending),
                _ => None,
            })
            .collect();
        assert_eq!(waits, vec![stages as u32 - 2], "wait for all but the youngest groups");
    }
}

/// The ring that leaves a trip's products in flight keeps the result too,
/// and its bookkeeping is as documented: production lags a trip (a slot is
/// refilled two trips after it was consumed), `stages - 3` copy groups stay
/// pending at a consume, every consume ends waiting for all but its
/// youngest chain and the loop ends waiting for every chain.
#[test_case(3; "three stages")]
#[test_case(4; "four stages")]
fn products_in_flight_preserve_the_result(stages: usize) {
    let (m, n, k) = (64usize, 32usize, 160usize);
    let prog = super::gemm_nt(m, n, k, 32, 32, 32, stages);
    let want = run(&prog, operands(m, n, k), &[("b", 1)]).unwrap();
    let prog = expand(prog, Fill::CpAsync, Core::Group, SyncCaps::default());
    let got = run(&prog, operands(m, n, k), &[("b", 1)]).unwrap();
    assert_eq!(got[2], want[2]);
    let syncs: Vec<Sync> =
        prog.walk().filter_map(|(_, s)| if let Stmt::Sync(s) = s { Some(*s) } else { None }).collect();
    let copies = syncs.iter().filter(|s| matches!(s, Sync::WaitAsync { .. })).count();
    assert_eq!(copies, 1);
    assert!(syncs.contains(&Sync::WaitAsync { pending: stages as u32 - 3 }), "{syncs:?}");
    let waits: Vec<u32> = syncs
        .iter()
        .filter_map(|s| if let Sync::WaitProducts { pending } = s { Some(*pending) } else { None })
        .collect();
    assert_eq!(waits, [1, 0], "one in the loop, one after it");
    let at = prog.body.0.iter().position(|s| matches!(s, Stmt::Loop(_))).expect("the ring's loop");
    let Stmt::Loop(l) = &prog.body.0[at] else { unreachable!() };
    assert!(matches!(prog.body.0[at + 1], Stmt::Sync(Sync::WaitProducts { pending: 0 })), "waited right after");
    let Stmt::If { then, .. } = &l.body.0[0] else { panic!("the consume first") };
    assert!(
        matches!(then.0.last(), Some(Stmt::Sync(Sync::WaitProducts { pending: 1 }))),
        "the consume ends in its wait"
    );
    // The produce stage's step is `iv - 1`: a lag of one trip.
    let Stmt::If { pred, .. } = &l.body.0[1] else { panic!("the produce second") };
    let Scalar::Bin(BinOp::And, started, _) = prog.scalar(*pred) else { panic!("started and before the end") };
    assert!(matches!(prog.scalar(*started), Scalar::Bin(BinOp::Le, _, iv) if *iv == l.iv));
}

/// Scalar `id` of `prog` where induction `iv` holds `at`: the arithmetic a
/// template's slots and phases are.
fn scalar_at(prog: &Program<Expanded>, id: ScalarId, iv: ScalarId, at: i64) -> i64 {
    let eval = |id| scalar_at(prog, id, iv, at);
    match prog.scalar(id) {
        Scalar::Const(v) => *v,
        Scalar::Induction if id == iv => at,
        // The block at the grid's origin.
        Scalar::Special(_) => 0,
        Scalar::Bin(op, a, b) => match (op, eval(*a), eval(*b)) {
            (BinOp::Add, a, b) => a + b,
            (BinOp::Sub, a, b) => a - b,
            (BinOp::Mul, a, b) => a * b,
            (BinOp::Min, a, b) => a.min(b),
            (BinOp::Max, a, b) => a.max(b),
            (BinOp::Div, a, b) => a / b,
            (BinOp::Rem, a, b) => a % b,
            (BinOp::Le, a, b) => i64::from(a <= b),
            (BinOp::Lt, a, b) => i64::from(a < b),
            (BinOp::And, a, b) => a & b,
            (op, ..) => panic!("{op:?} in a slot or a phase"),
        },
        other => panic!("{other:?} in a slot or a phase"),
    }
}

/// A ring filled in bulk on a target with slot barriers: the result is the
/// ring's, every slot's barrier is made ahead of the loop, a trip's fill
/// arrives at its slot's barrier with the bytes of both operands and no
/// thread commits or waits for a copy group.
#[test_case(3; "three stages")]
#[test_case(4; "four stages")]
fn a_bulk_ring_fills_on_its_slots_barriers(stages: usize) {
    let (m, n, k) = (64usize, 32usize, 160usize);
    let cfg = |schedule| GemmCfg { schedule, ..super::gemm_nt_cfg([32, 32, 32], stages) };
    let prog = |schedule| gemm::<BF16>(&assumed(super::gemm_spec(m, n, k), cfg(schedule)));
    let want = run(&prog(Schedule::ring(stages, true)), operands(m, n, k), &[("b", 1)]).unwrap();
    let slots = SyncCaps { slot_barrier: true, ..SyncCaps::default() };
    let bulk = expand(prog(Schedule::bulk(stages, true)), Fill::CpAsync, Core::Group, slots);
    assert_eq!(run(&bulk, operands(m, n, k), &[("b", 1)]).unwrap()[2], want[2]);
    let syncs: Vec<Sync> =
        bulk.walk().filter_map(|(_, s)| if let Stmt::Sync(s) = s { Some(*s) } else { None }).collect();
    assert_eq!(syncs.iter().filter(|s| matches!(s, Sync::SlotInit { .. })).count(), stages);
    assert!(syncs.iter().any(|s| matches!(s, Sync::SlotArrive { bytes, .. } if *bytes as usize == (32 + 32) * 32 * 2)));
    assert!(!syncs.iter().any(|s| matches!(s, Sync::CommitAsync | Sync::WaitAsync { .. })), "{syncs:?}");
    let modes = bulk.walk().filter_map(|(_, s)| if let Stmt::Copy { mode, .. } = s { Some(*mode) } else { None });
    assert_eq!(modes.filter(|mode| matches!(mode, CopyMode::Bulk { .. })).count(), 2, "both operands");
    // Without slot barriers the schedule is the ring it names.
    let plain = expand(prog(Schedule::bulk(stages, true)), Fill::CpAsync, Core::Group, SyncCaps::default());
    let ring = expand(prog(Schedule::ring(stages, true)), Fill::CpAsync, Core::Group, SyncCaps::default());
    assert_eq!(plain.body, ring.body);
}

/// A two-warpgroup attention over `trips` key blocks, its ring filled in
/// bulk, expanded for a target with slot barriers.
fn left_ring(stages: usize, trips: usize, mask: AttnMask) -> Program<Expanded> {
    let cfg = FaCfg { core: Core::Group, schedule: Schedule::bulk(stages, false), ..FaCfg::new(128, 32, stages) };
    let (batch, scale) = (crate::kernels::Batch::Static(1), 0.125);
    let (t, tk) = (128, 32 * trips);
    let spec = AttnSpec { batch, t, tk, heads: 2, kv_heads: 1, d: 64, mask, cache: None, packed: false, scale };
    let slots = SyncCaps { slot_barrier: true, ..SyncCaps::default() };
    expand(attention::<BF16>(&assumed(spec, cfg)), Fill::CpAsync, Core::Group, slots)
}

fn steady(prog: &Program<Expanded>) -> &Loop<Expanded> {
    let mut loops = prog.walk().filter_map(|(_, s)| if let Stmt::Loop(l) = s { Some(l) } else { None });
    loops.find(|l| l.trip == Some(Trip::Steady)).expect("the ring")
}

fn syncs_of(block: &Block<Expanded>) -> Vec<Sync> {
    block.walk().filter_map(|(_, s)| if let Stmt::Sync(s) = s { Some(*s) } else { None }).collect()
}

/// A ring some of whose copies stay their threads' chunks keeps the block
/// barrier and the copy groups beside the slots' fills: no slot is left.
#[test]
fn a_ring_of_mixed_fills_keeps_its_block_barrier() {
    let prog = left_ring(3, 4, AttnMask { bias: Some(Bias::PerBatch), key_lens: true, ..AttnMask::default() });
    let syncs = syncs_of(&steady(&prog).body);
    let has = |want: fn(&Sync) -> bool| syncs.iter().any(want);
    assert!(has(|s| matches!(s, Sync::SlotWait { .. })) && has(|s| matches!(s, Sync::WaitAsync { .. })), "{syncs:?}");
    assert!(has(|s| matches!(s, Sync::Barrier { .. })) && has(|s| matches!(s, Sync::CommitAsync)), "{syncs:?}");
    assert!(!has(|s| matches!(s, Sync::SlotLeave { .. } | Sync::SlotVacant { .. })), "{syncs:?}");
}

/// A ring filled in bulk inside a loop stays its threads' chunks: its slot
/// barriers are initialized once, and initializing one again while a
/// lagging warp still arrives at it is undefined.
#[test_case(false => true; "once a block")]
#[test_case(true => false; "inside a loop")]
fn a_bulk_ring_runs_once_a_block(nested: bool) -> bool {
    let stages = 3;
    let cfg = GemmCfg { schedule: Schedule::bulk(stages, true), ..super::gemm_nt_cfg([32, 32, 32], stages) };
    let mut prog = gemm::<BF16>(&assumed(super::gemm_spec(64, 32, 160), cfg));
    if nested {
        let at = prog.body.0.iter().position(|s| matches!(s, Stmt::Pipeline(_))).expect("the ring");
        let ring = prog.body.0.remove(at);
        let mut scalar = |s: Scalar| {
            prog.scalars.push(s);
            ScalarId(prog.scalars.len() as u32 - 1)
        };
        let (iv, start, extent) = (scalar(Scalar::Induction), scalar(Scalar::Const(0)), scalar(Scalar::Const(2)));
        let body = Block(vec![ring]);
        prog.body.0.insert(at, Stmt::Loop(Loop { iv, start, extent, carried: vec![], body, unroll: 1, trip: None }));
    }
    let slots = SyncCaps { slot_barrier: true, ..SyncCaps::default() };
    let out = expand(prog, Fill::CpAsync, Core::Group, slots);
    count(&out, |s| matches!(s, Stmt::Sync(Sync::SlotInit { .. }))) == stages
}

/// The ring's first pipeline, wherever it is nested.
fn pipeline_of(block: &mut Block) -> Option<&mut Pipeline> {
    block.0.iter_mut().find_map(|stmt| match stmt {
        Stmt::Pipeline(p) => Some(p),
        Stmt::Loop(l) => pipeline_of(&mut l.body),
        Stmt::Warps { body } => pipeline_of(body),
        Stmt::If { then, otherwise, .. } => pipeline_of(then).or_else(|| pipeline_of(otherwise)),
        _ => None,
    })
}

/// Two warpgroups leave a slot only where products alone read it: the
/// warp's wait for them covers every lane, where a lane's own read of the
/// slot is not covered by its first lane's arrival, and keeps the block
/// barrier.
#[test_case(false => true; "products read the slots")]
#[test_case(true => false; "a lane reads a slot")]
fn a_slot_is_left_where_products_alone_read_it(lanes_read: bool) -> bool {
    let cfg = FaCfg { core: Core::Group, schedule: Schedule::bulk(3, false), ..FaCfg::new(128, 32, 3) };
    let (batch, scale) = (crate::kernels::Batch::Static(1), 0.125);
    let mask = AttnMask::default();
    let spec =
        AttnSpec { batch, t: 128, tk: 128, heads: 2, kv_heads: 1, d: 64, mask, cache: None, packed: false, scale };
    let mut prog = attention::<BF16>(&assumed(spec, cfg));
    if lanes_read {
        let smem = |prog: &Program, v: ValId| matches!(prog.value(v).place, Place::Smem { .. });
        let consume = pipeline_of(&mut prog.body).expect("the ring").consume.body.clone();
        let keys = consume.walk().find_map(|(_, stmt)| match stmt {
            Stmt::Let { op: TileOp::Mma { b, .. }, .. } if smem(&prog, *b) => Some(*b),
            _ => None,
        });
        let keys = keys.expect("a product of the keys' slot");
        let Value { dtype, shape, .. } = prog.value(keys).clone();
        prog.values.push(Value { dtype, shape, place: Place::Reg });
        let dst = ValId(prog.values.len() as u32 - 1);
        let read = Stmt::Copy { dst, src: keys, mode: CopyMode::Sync, scope: Scope::Block };
        pipeline_of(&mut prog.body).expect("the ring").consume.body.0.push(read);
    }
    let slots = SyncCaps { slot_barrier: true, ..SyncCaps::default() };
    let out = expand(prog, Fill::CpAsync, Core::Group, slots);
    count(&out, |s| matches!(s, Stmt::Sync(Sync::SlotLeave { .. }))) > 0
}

proptest! {
    /// Two warpgroups over a ring filled in bulk alone meet at no block
    /// barrier inside it: each leaves the slot it read on a barrier of the
    /// slot's own, apart from the fills', and a slot's refill waits for the
    /// parity of the phase its last readers completed (the phase before
    /// the first passes: nobody has read the slot).
    #[test]
    fn a_refill_waits_for_its_slots_last_readers(stages in 2usize..=4, trips in 1usize..=16) {
        let prog = left_ring(stages, trips, AttnMask::default());
        let l = steady(&prog);
        let [Stmt::If { pred: consuming, then: consume, .. }, Stmt::If { pred: producing, then: produce, .. }] =
            &l.body.0[..] else { panic!("a consume and a produce") };
        let Some(Stmt::Sync(Sync::SlotWait { bar: full, .. })) = consume.0.first() else { panic!("the fill's wait") };
        let Some(Stmt::Sync(Sync::SlotLeave { bar: read })) = consume.0.last() else { panic!("the leaving") };
        let [Stmt::Sync(Sync::SlotVacant { bar: refilled, phase }), Stmt::Sync(Sync::SlotArrive { bar: filled, .. })] =
            &produce.0[..2] else { panic!("the refill's wait, then its arrival") };
        let syncs = syncs_of(&l.body);
        let fenced = |s: &Sync| matches!(s, Sync::Barrier { .. } | Sync::CommitAsync | Sync::WaitAsync { .. });
        prop_assert!(!syncs.iter().any(fenced), "{:?}", syncs);
        let opened = prog.walk().filter(|(_, s)| matches!(s, Stmt::Sync(Sync::SlotOpen { .. }))).count();
        prop_assert_eq!((opened, prog.smem[read.bars.index()].elems), (stages, 2 * stages));
        let at_iv = |id: ScalarId, at: i64| scalar_at(&prog, id, l.iv, at);
        // Per slot: the steps read and left, and the fills issued.
        let mut slots = vec![(0i64, 0i64); stages];
        for at in 0..at_iv(l.extent, 0) {
            if at_iv(*consuming, at) != 0 {
                let slot = at_iv(full.slot, at) as usize;
                prop_assert_eq!(at_iv(read.slot, at) as usize, stages + slot, "a slot's two barriers");
                slots[slot].0 += 1;
            }
            if at_iv(*producing, at) != 0 {
                let slot = at_iv(filled.slot, at) as usize;
                prop_assert_eq!(at_iv(refilled.slot, at) as usize, stages + slot, "a slot's two barriers");
                let (left, issued) = &mut slots[slot];
                prop_assert_eq!(*left, *issued, "every fill before this one was read");
                prop_assert_eq!(at_iv(*phase, at), (*left + 1) % 2, "the parity of the last phase completed");
                *issued += 1;
            }
        }
        prop_assert_eq!(slots.iter().map(|s| s.0).sum::<i64>() as usize, trips);
    }

    /// A slot's barrier completes a phase a fill, the phases alternating
    /// from 0: every consume waits for the parity of the one fill its slot
    /// has taken since the consume before it, which was issued a trip
    /// earlier at least, and the slot is refilled only after.
    #[test]
    fn a_consume_waits_for_its_slots_latest_fill(stages in 3usize..=6, trips in 1usize..=24) {
        let (m, n, k) = (64, 32, 32 * trips);
        let cfg = GemmCfg { schedule: Schedule::bulk(stages, false), ..super::gemm_nt_cfg([32, 32, 32], stages) };
        let slots = SyncCaps { slot_barrier: true, ..SyncCaps::default() };
        let prog = expand(gemm::<BF16>(&assumed(super::gemm_spec(m, n, k), cfg)), Fill::CpAsync, Core::Group, slots);
        let Some(Stmt::Loop(l)) = prog.body.0.iter().find(|s| matches!(s, Stmt::Loop(_))) else { panic!("the ring") };
        let [Stmt::If { pred: consuming, then: consume, .. }, Stmt::If { pred: producing, then: produce, .. }] =
            &l.body.0[..] else { panic!("a consume and a produce") };
        let Some(Stmt::Sync(Sync::SlotWait { bar: waited, phase })) = consume.0.first() else { panic!("the wait") };
        let Some(Stmt::Sync(Sync::SlotArrive { bar: filled, .. })) = produce.0.first() else { panic!("the arrival") };
        let holds = |pred: ScalarId, at: i64| scalar_at(&prog, pred, l.iv, at) != 0;
        // Per slot: the fills issued and the fills consumed.
        let mut fills = vec![(0i64, 0i64); stages];
        let extent = scalar_at(&prog, l.extent, l.iv, 0);
        prop_assert_eq!(extent as usize, trips + stages - 1);
        for at in 0..extent {
            if holds(*consuming, at) {
                let slot = scalar_at(&prog, waited.slot, l.iv, at) as usize;
                let (issued, consumed) = &mut fills[slot];
                prop_assert_eq!(*issued, *consumed + 1, "one fill since the last consume");
                prop_assert_eq!(scalar_at(&prog, *phase, l.iv, at), *consumed % 2, "its parity");
                *consumed += 1;
            }
            if holds(*producing, at) {
                let (issued, consumed) = &mut fills[scalar_at(&prog, filled.slot, l.iv, at) as usize];
                prop_assert_eq!(*issued, *consumed, "a slot is refilled once consumed");
                *issued += 1;
            }
        }
        prop_assert_eq!(fills.iter().map(|f| f.1).sum::<i64>() as usize, trips);
    }
}

/// The two-crew expansion keeps the GEMM's result (the interpreter runs the
/// crews as one, its barriers and priorities no-ops) and leaves no pipeline:
/// a head trip, the second crew's extra barrier, the steady loop, the first
/// crew's extra barrier; a trip holds two barriers per product.
#[test_case(64, 4; "four slices")]
#[test_case(32, 2; "two slices")]
fn ping_pong_expansion_preserves_the_result(bk: usize, slices: usize) {
    let (m, n, k) = (64usize, 64usize, 256usize);
    let mut prog = super::gemm_nt_crews(m, n, k, [64, 64, bk]);
    let want = run(&prog, operands(m, n, k), &[("b", 1)]).unwrap();
    // As the lowering: a product's shared operands become explicit reads first.
    crate::lower::materialize_operands(&mut prog, Core::Warp);
    let prog = expand(prog, Fill::Staged, Core::Warp, SyncCaps::default());
    let got = run(&prog, operands(m, n, k), &[("b", 1)]).unwrap();
    assert_eq!(got[2], want[2]);
    assert_eq!(count(&prog, |s| matches!(s, Stmt::Pipeline(_))), 0);
    assert_eq!(count(&prog, |s| matches!(s, Stmt::Loop(_))), 2, "a head trip and the steady loop");
    let barriers = count(&prog, |s| matches!(s, Stmt::Sync(Sync::Barrier { .. })));
    assert_eq!(barriers, 1 + 2 + 2 * slices, "the head's, the stagger pair, two per product");
    assert_eq!(count(&prog, |s| matches!(s, Stmt::Sync(Sync::Prio { level: 3 }))), slices);
}

/// A two-crew trip is clusters: the fragment reads of a product (the last
/// product's a cluster early), a barrier, the product at raised priority, a
/// barrier; the next step's loads lead the load clusters, each operand's
/// together (A's in the first, B's in the one before the last), and the
/// commit, in issue order, sits alone in the last; the stagger barriers are
/// one `if` before the loop for the second crew and one after it for the
/// first.
#[test]
fn ping_pong_cuts_a_trip_into_clusters() {
    let mut prog = super::gemm_nt_crews(64, 64, 256, [64, 64, 64]);
    crate::lower::materialize_operands(&mut prog, Core::Warp);
    let prog = expand(prog, Fill::Staged, Core::Warp, SyncCaps::default());
    let kind = |s: &Stmt<Expanded>| match s {
        Stmt::Copy { dst, src, mode: CopyMode::Staged, .. } => match (&prog.value(*src).place, prog.value(*dst).tier())
        {
            (Place::Global { param, .. }, Tier::Reg) => ["issue a", "issue b"][param.index()],
            (_, Tier::Smem) => "commit",
            (place, tier) => panic!("{place:?} {tier:?}"),
        },
        Stmt::Copy { dst, src, mode: CopyMode::Sync, .. } => match (prog.value(*src).tier(), prog.value(*dst).tier()) {
            (Tier::Smem, Tier::Reg) => "read",
            tiers => panic!("{tiers:?}"),
        },
        Stmt::Let { op: TileOp::Mma { .. }, .. } => "mma",
        Stmt::Let { .. } => "compute",
        Stmt::Sync(Sync::Barrier { .. }) => "barrier",
        Stmt::Sync(Sync::Prio { level: 3 }) => "prio",
        Stmt::Sync(Sync::Prio { level: 0 }) => "unprio",
        other => panic!("{other:?}"),
    };
    let at = prog.body.0.iter().position(|s| matches!(s, Stmt::Loop(_))).expect("the head trip");
    let [Stmt::Loop(head), Stmt::If { then: wait, .. }, Stmt::Loop(steady), Stmt::If { then: realign, .. }] =
        &prog.body.0[at..at + 4]
    else {
        panic!("{:?}", prog.body.0.len())
    };
    assert_eq!(
        head.body.0.iter().map(kind).collect::<Vec<_>>(),
        [["issue a", "issue b"].repeat(4), vec!["commit"; 8], vec!["barrier"]].concat()
    );
    assert_eq!((wait.0.len(), realign.0.len()), (1, 1), "one barrier each");
    assert!(matches!(
        (&wait.0[0], &realign.0[0]),
        (Stmt::Sync(Sync::Barrier { .. }), Stmt::Sync(Sync::Barrier { .. }))
    ));
    let kinds: Vec<&str> = steady.body.0.iter().map(kind).collect();
    let cluster = |issues: &[&'static str], reads: usize, commits: usize| -> Vec<&'static str> {
        [
            issues.to_vec(),
            vec!["read"; reads],
            vec!["commit"; commits],
            vec!["barrier", "prio", "mma", "unprio", "barrier"],
        ]
        .concat()
    };
    let want = [cluster(&["issue a"; 4], 2, 0), cluster(&[], 2, 0), cluster(&["issue b"; 4], 4, 0), cluster(&[], 0, 8)]
        .concat();
    assert_eq!(kinds, want);
    // Each commit stores the registers of the load issued in its place.
    let staged = |want: Tier| -> Vec<ValId> {
        let regs = steady.body.0.iter().filter_map(|s| match s {
            Stmt::Copy { dst, src, mode: CopyMode::Staged, .. } if prog.value(*src).tier() == want => {
                Some(if want == Tier::Global { *dst } else { *src })
            }
            _ => None,
        });
        regs.collect()
    };
    assert_eq!(staged(Tier::Global), staged(Tier::Reg));
    assert_eq!(steady.unroll, 1);
    assert_eq!(steady.carried.len(), 1);
}

/// A copy carries the scope it was recorded in, and the halves register
/// staging splits it into keep it: inside a `Warps` statement every copy is
/// one warp's own, outside it the block's.
#[test]
fn staged_halves_keep_their_copys_scope() {
    use crate::build::BF16;
    use crate::kernels::attention::{AttnMask, FaCfg, attention};
    let cfg = FaCfg { key_warps: 4, ..FaCfg::new(16, 16, 2) };
    let spec = assumed(super::attention::probe_spec(1, 1, 1, 64, 64, AttnMask::default()), cfg);
    let prog = expand(attention::<BF16>(&spec), Fill::Staged, Core::Warp, SyncCaps::default());
    fn copies(block: &Block<Expanded>, inside: bool, out: &mut Vec<(bool, Scope, CopyMode)>) {
        for s in &block.0 {
            if let Stmt::Copy { mode, scope, .. } = s {
                out.push((inside, *scope, *mode));
            }
            let inside = inside || matches!(s, Stmt::Warps { .. });
            s.blocks().for_each(|b| copies(b, inside, out));
        }
    }
    let mut seen = vec![];
    copies(&prog.body, false, &mut seen);
    assert!(
        seen.iter().any(|(inside, _, mode)| *inside && *mode == CopyMode::Staged),
        "staged halves inside the warps"
    );
    assert!(seen.iter().any(|(inside, ..)| !inside), "copies outside the warps");
    for (inside, scope, mode) in seen {
        let want = if inside { Scope::Warp } else { Scope::Block };
        assert_eq!(scope, want, "a {mode:?} copy {}", if inside { "inside the warps" } else { "of the block" });
    }
}

/// Register staging splits every global→shared copy into a load before the
/// consumer and a store after it, and peels the ring into three loops over
/// one induction variable: the first trip loads and commits step 0, a
/// steady trip loads, consumes and ends in `tail` with no branch in its
/// body, and the last trip only consumes what the steady loop left. A
/// trip's loads are fenced ahead of its products. Two slots fence the
/// commit and unroll over the slots; one slot commits between two barriers
/// (the slot is read out before it is overwritten, then seen), its slot
/// scalars the constant 0 and nothing to unroll.
#[test_case(2, ["fence", "commit", "commit", "barrier"], 2; "two slots fence the commit")]
#[test_case(1, ["barrier", "commit", "commit", "barrier"], 1; "one slot commits between two barriers")]
fn register_staging_peels_the_ring_into_three_branch_free_loops(stages: usize, tail: [&str; 4], unroll: u32) {
    let prog = super::gemm_nt(32, 32, 64, 32, 32, 32, stages);
    let slots: Vec<ScalarId> = prog
        .walk()
        .filter_map(|(_, s)| if let Stmt::Pipeline(p) = s { Some([p.produce.slot, p.consume.slot]) } else { None })
        .flatten()
        .collect();
    let schedule = prog.walk().find_map(|(_, s)| if let Stmt::Pipeline(p) = s { Some(p.schedule) } else { None });
    let prog = expand(prog, Fill::Staged, Core::Warp, SyncCaps::default());
    let constant = slots.iter().all(|&s| *prog.scalar(s) == Scalar::Const(0));
    assert!(!slots.is_empty() && constant == (stages == 1), "slot scalars constant: {constant}");
    let loops: Vec<&Loop<Expanded>> =
        prog.body.0.iter().filter_map(|s| if let Stmt::Loop(l) = s { Some(l) } else { None }).collect();
    let [head, steady, drain] = loops[..] else { panic!("three loops, got {}", loops.len()) };
    let kind = |s: &Stmt<Expanded>| match s {
        Stmt::Copy { dst, src, mode: CopyMode::Staged, .. } => match (prog.value(*src).tier(), prog.value(*dst).tier())
        {
            (Tier::Global, Tier::Reg) => "issue",
            (Tier::Reg, Tier::Smem) => "commit",
            tiers => panic!("{tiers:?}"),
        },
        Stmt::Let { .. } => "compute",
        Stmt::Sync(Sync::Fence) => "fence",
        Stmt::Sync(Sync::Barrier { .. }) => "barrier",
        other => panic!("{other:?}"),
    };
    let kinds = |l: &Loop<Expanded>| l.body.0.iter().map(kind).collect::<Vec<_>>();
    assert_eq!(kinds(head), ["issue", "issue", "commit", "commit", "barrier"]);
    let k = kinds(steady);
    let computes = k.len() - 7;
    assert_eq!(k[..3], ["issue", "issue", "fence"]);
    assert!(computes > 0 && k[3..3 + computes].iter().all(|k| *k == "compute"), "{k:?}");
    assert_eq!(k[3 + computes..], tail);
    assert_eq!(kinds(drain), vec!["compute"; computes]);
    assert_eq!((head.unroll, steady.unroll, drain.unroll), (1, unroll, 1));
    assert_eq!(schedule.map(Schedule::per_iteration), Some(unroll as usize), "a steady iteration's trips");
    assert_eq!(count(&prog, |s| matches!(s, Stmt::Sync(Sync::Fence))), stages, "fences");
    assert!(head.iv == steady.iv && steady.iv == drain.iv, "one induction variable");
    assert!(head.carried.is_empty());
    assert_eq!(steady.carried.len(), 1);
    let (s, d) = (steady.carried[0], drain.carried[0]);
    assert_eq!((d.init, d.phi, d.next), (s.phi, s.phi, s.next), "the drain continues the steady accumulator");
}

/// A fill builds only the rings its template runs: through registers one
/// or two slots, so a three-slot register-staged ring (which the template
/// refused with a panic) is `None` before any program holds it, and every
/// config a family's lattice spans on a target runs on that target's fill.
#[test]
fn a_fill_builds_only_the_rings_it_runs() {
    assert_eq!(Fill::Staged.ring(3, false), None);
    assert_eq!(Fill::Staged.ring(2, true), Some(Schedule::ring(2, true)));
    assert_eq!(Fill::CpAsync.ring(3, false), Some(Schedule::ring(3, false)));
    assert_eq!(Fill::CpAsync.ring(0, false), None);
    for (name, target) in crate::sweep::targets() {
        let fill = target.fill;
        let gemm = crate::ops::config::gemm_lattice(&target).into_iter().map(|c| c.schedule);
        let attention =
            crate::ops::config::AttentionProblem { tiles: 1, t: 1024, tk: 1024, d: 64, causal: false, bias: false }
                .lattice(&target)
                .into_iter()
                .map(|c| c.schedule);
        let nearest = super::self_join(4096, 4096, 128, crate::kernels::nearest::Metric::L2, 8)
            .lattice(&target)
            .into_iter()
            .map(|c| c.schedule);
        for schedule in gemm.chain(attention).chain(nearest) {
            assert!(fill.runs(schedule), "{name}: {schedule:?} on {fill:?}");
        }
    }
}

/// A nearest kernel of `k` scores a row over a register-staged ring of
/// `stages` slots, lane queues two wide and four deep, as the k = 16 picks.
fn listing(k: usize, stages: usize) -> Program {
    use crate::kernels::nearest::{LaneQueue, Metric, NearestCfg, Pow2, nearest};
    let lanes = LaneQueue { width: Pow2::of(2), depth: Pow2::of(4), passes: true };
    let cfg = NearestCfg { warp_rows: 16, lanes: Some(lanes), ..NearestCfg::new(64, 16, stages) };
    let spec = super::nearest::plain(64, 64, 32, Metric::NegDot, k, ScalarDType::BFloat16, cfg);
    nearest::<crate::build::BF16>(&spec)
}

/// Where a block barrier's halves part, a two-slot register-staged trip
/// whose consume ends in branches past its last shared read (a nearest
/// list's vote and pushes) commits and arrives before them and waits after
/// them, its lists the authored ones; one slot, a target whose halves do not
/// part, or a tail of products alone (a GEMM's) keep the whole barrier.
#[test_case(16, 2, true, true; "lists over two slots part")]
#[test_case(16, 1, true, false; "one slot keeps the barrier")]
#[test_case(16, 2, false, false; "unsplit target keeps the barrier")]
#[test_case(0, 2, true, false; "products alone keep the barrier")]
fn a_branching_tail_runs_between_the_barriers_halves(k: usize, stages: usize, split: bool, parted: bool) {
    let authored = if k == 0 { super::gemm_nt(64, 32, 128, 32, 32, 32, stages) } else { listing(k, stages) };
    let params: Vec<Vec<f64>> = authored
        .params
        .iter()
        .enumerate()
        .map(|(p, param)| (0..param.elems).map(|i| ((i * 7 + p * 5) % 23) as f64 / 8.0 - 1.0).collect())
        .collect();
    let want = run(&authored, params.clone(), &[]).unwrap();
    let prog = expand(authored, Fill::Staged, Core::Warp, SyncCaps { split_barrier: split, ..SyncCaps::default() });
    assert_eq!(run(&prog, params, &[]).unwrap(), want, "the template moved a read past the commit");
    let halves = |s: &Stmt<Expanded>| matches!(s, Stmt::Sync(Sync::Arrive | Sync::Await));
    if !parted {
        assert_eq!(count(&prog, halves), 0, "the whole barrier");
        return;
    }
    // Unmaterialized, a product reads its shared operand itself.
    let shared = |v: ValId| prog.value(v).tier() == Tier::Smem;
    let reads_shared = |s: &Stmt<Expanded>| match s {
        Stmt::Copy { src, .. } => shared(*src),
        Stmt::Let { op, .. } => op.srcs().any(shared),
        _ => false,
    };
    let commits = |s: &Stmt<Expanded>| matches!(s, Stmt::Copy { mode: CopyMode::Staged, dst, .. } if prog.value(*dst).tier() == Tier::Smem);
    // The steady trip: the loop that both reads its slot and commits the next.
    let steady = prog.walk().find_map(|(_, s)| match s {
        Stmt::Loop(l) if l.body.0.iter().any(reads_shared) && l.body.0.iter().any(commits) => Some(&l.body.0),
        _ => None,
    });
    let body = steady.expect("a steady trip");
    let at = |want: &dyn Fn(&Stmt<Expanded>) -> bool| body.iter().position(want).expect("in the trip");
    let commit = body.iter().rposition(commits).expect("a commit");
    let arrive = at(&|s| matches!(s, Stmt::Sync(Sync::Arrive)));
    let branch = at(&|s| matches!(s, Stmt::If { .. }));
    assert!(commit < arrive && arrive < branch, "commit, arrive, then the branches");
    assert!(matches!(body.last(), Some(Stmt::Sync(Sync::Await))), "the trip ends waiting");
    assert_eq!(count(&prog, halves), 2, "one arrive and one await a trip");
    assert!(!body[arrive..].iter().any(reads_shared), "no shared read past the arrive");
}
