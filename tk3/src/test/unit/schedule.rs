use svod_dtype::ScalarDType;
use test_case::test_case;

use crate::interp::{round_to, run};
use crate::ir::*;
use crate::schedule::{Prefetch, Schedule, expand};

fn count(prog: &Program, pred: impl Fn(&Stmt) -> bool) -> usize {
    prog.walk().filter(|(_, s)| pred(s)).count()
}

/// Expanding the pipeline keeps the GEMM's result (the interpreter treats
/// synchronization as no-ops, so this checks the step/slot bookkeeping and
/// the prologue and drain, guarded or peeled) and leaves no pipeline behind.
#[test_case(Prefetch::CpAsync, 1; "cp.async one stage")]
#[test_case(Prefetch::CpAsync, 2; "cp.async two stages")]
#[test_case(Prefetch::CpAsync, 4; "cp.async four stages")]
#[test_case(Prefetch::RegisterStaged, 2; "register staged")]
#[test_case(Prefetch::RegisterStaged, 1; "register staged, one slot")]
fn expansion_preserves_the_result(prefetch: Prefetch, stages: usize) {
    let (m, n, k) = (64usize, 32usize, 160usize);
    let mut prog = super::gemm_nt(m, n, k, 32, 32, 32, stages);
    let a: Vec<f64> = (0..m * k).map(|i| round_to(ScalarDType::BFloat16, ((i * 13) % 17) as f64 / 8.0 - 1.0)).collect();
    let b: Vec<f64> = (0..n * k).map(|i| round_to(ScalarDType::BFloat16, ((i * 7) % 19) as f64 / 8.0 - 1.0)).collect();
    let want = run(&prog, vec![a.clone(), b.clone(), vec![0.0; m * n]], &[("b", 1)]).unwrap();
    expand(&mut prog, Schedule::Uniform { prefetch, unroll: true, async_products: false });
    let got = run(&prog, vec![a, b, vec![0.0; m * n]], &[("b", 1)]).unwrap();
    assert_eq!(got[2], want[2]);
    assert_eq!(count(&prog, |s| matches!(s, Stmt::Pipeline(_))), 0);
    let staged = prefetch == Prefetch::RegisterStaged;
    assert_eq!(count(&prog, |s| matches!(s, Stmt::Loop(_))), if staged { 3 } else { 1 }, "peeled or one loop");
    let barriers = count(&prog, |s| matches!(s, Stmt::Sync(Sync::Barrier { .. })));
    let want = if staged && stages == 1 {
        3
    } else if staged || (prefetch == Prefetch::CpAsync && stages == 1) {
        2
    } else {
        1
    };
    assert_eq!(barriers, want, "one per trip (two around a one-slot commit), plus the peeled prologue's");
    if prefetch == Prefetch::CpAsync && stages >= 2 {
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
    let mut prog = super::gemm_nt(m, n, k, 32, 32, 32, stages);
    let a: Vec<f64> = (0..m * k).map(|i| round_to(ScalarDType::BFloat16, ((i * 13) % 17) as f64 / 8.0 - 1.0)).collect();
    let b: Vec<f64> = (0..n * k).map(|i| round_to(ScalarDType::BFloat16, ((i * 7) % 19) as f64 / 8.0 - 1.0)).collect();
    let want = run(&prog, vec![a.clone(), b.clone(), vec![0.0; m * n]], &[("b", 1)]).unwrap();
    expand(&mut prog, Schedule::Uniform { prefetch: Prefetch::CpAsync, unroll: true, async_products: true });
    let got = run(&prog, vec![a, b, vec![0.0; m * n]], &[("b", 1)]).unwrap();
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

/// A copy carries the scope it was recorded in, and the halves register
/// staging splits it into keep it: inside a `Warps` statement every copy is
/// one warp's own, outside it the block's.
#[test]
fn staged_halves_keep_their_copys_scope() {
    use crate::build::BF16;
    use crate::kernels::Batch;
    use crate::kernels::attention::{AttnMask, AttnSpec, FaCfg, attention};
    let cfg = FaCfg { key_warps: 4, ..FaCfg::new(16, 16, 2) };
    let mask = AttnMask::default();
    let spec = AttnSpec {
        batch: Batch::Static(1),
        t: 1,
        tk: 64,
        heads: 1,
        kv_heads: 1,
        d: 64,
        mask,
        cache: None,
        scale: 0.125,
        cfg,
    };
    let mut prog = attention::<BF16>(&spec);
    expand(&mut prog, Schedule::Uniform { prefetch: Prefetch::RegisterStaged, unroll: false, async_products: false });
    fn copies(block: &Block, inside: bool, out: &mut Vec<(bool, Scope, CopyMode)>) {
        for s in &block.0 {
            match s {
                Stmt::Copy { mode, scope, .. } => out.push((inside, *scope, *mode)),
                Stmt::Loop(l) => copies(&l.body, inside, out),
                Stmt::Warps { body } => copies(body, true, out),
                Stmt::If { then, otherwise, .. } => {
                    copies(then, inside, out);
                    copies(otherwise, inside, out);
                }
                Stmt::Pipeline(p) => {
                    copies(&p.produce.body, inside, out);
                    copies(&p.consume.body, inside, out);
                }
                _ => {}
            }
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
/// steady trip loads, consumes, fences, commits and barriers with no branch
/// in its body, and the last trip only consumes what the steady loop left.
#[test]
fn register_staging_peels_the_ring_into_three_branch_free_loops() {
    let mut prog = super::gemm_nt(32, 32, 64, 32, 32, 32, 2);
    expand(&mut prog, Schedule::Uniform { prefetch: Prefetch::RegisterStaged, unroll: true, async_products: false });
    let loops: Vec<&Loop> =
        prog.body.0.iter().filter_map(|s| if let Stmt::Loop(l) = s { Some(l) } else { None }).collect();
    let [head, steady, drain] = loops[..] else { panic!("three loops, got {}", loops.len()) };
    let kind = |s: &Stmt| match s {
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
    let kinds = |l: &Loop| l.body.0.iter().map(kind).collect::<Vec<_>>();
    assert_eq!(kinds(head), ["issue", "issue", "commit", "commit", "barrier"]);
    let k = kinds(steady);
    let computes = k.len() - 6;
    assert_eq!(k[..2], ["issue", "issue"]);
    assert!(computes > 0 && k[2..2 + computes].iter().all(|k| *k == "compute"), "{k:?}");
    assert_eq!(k[2 + computes..], ["fence", "commit", "commit", "barrier"]);
    assert_eq!(kinds(drain), vec!["compute"; computes]);
    assert_eq!((head.unroll, steady.unroll, drain.unroll), (1, 2, 1));
    assert!(head.iv == steady.iv && steady.iv == drain.iv, "one induction variable");
    assert!(head.carried.is_empty());
    assert_eq!(steady.carried.len(), 1);
    let (s, d) = (steady.carried[0], drain.carried[0]);
    assert_eq!((d.init, d.phi, d.next), (s.phi, s.phi, s.next), "the drain continues the steady accumulator");
}

/// Over one slot the steady trip's commit sits between two barriers (the
/// slot is read out before it is overwritten, then seen) where two slots
/// need a fence, both slot scalars are the constant 0, and the steady loop
/// is not unrolled: there is no slot arithmetic to fold.
#[test]
fn one_slot_staging_commits_between_two_barriers() {
    let mut prog = super::gemm_nt(32, 32, 64, 32, 32, 32, 1);
    let slots: Vec<ScalarId> = prog
        .walk()
        .filter_map(|(_, s)| if let Stmt::Pipeline(p) = s { Some([p.produce.slot, p.consume.slot]) } else { None })
        .flatten()
        .collect();
    expand(&mut prog, Schedule::Uniform { prefetch: Prefetch::RegisterStaged, unroll: true, async_products: false });
    assert!(!slots.is_empty() && slots.iter().all(|&s| *prog.scalar(s) == Scalar::Const(0)), "one slot");
    let loops: Vec<&Loop> =
        prog.body.0.iter().filter_map(|s| if let Stmt::Loop(l) = s { Some(l) } else { None }).collect();
    let [head, steady, drain] = loops[..] else { panic!("three loops, got {}", loops.len()) };
    let kind = |s: &Stmt| match s {
        Stmt::Copy { mode: CopyMode::Staged, .. } => "copy",
        Stmt::Let { .. } => "compute",
        Stmt::Sync(Sync::Barrier { .. }) => "barrier",
        other => panic!("{other:?}"),
    };
    let kinds = |l: &Loop| l.body.0.iter().map(kind).collect::<Vec<_>>();
    assert_eq!(kinds(head), ["copy", "copy", "copy", "copy", "barrier"]);
    let k = kinds(steady);
    let computes = k.len() - 6;
    assert_eq!(k[..2], ["copy", "copy"]);
    assert!(computes > 0 && k[2..2 + computes].iter().all(|k| *k == "compute"), "{k:?}");
    assert_eq!(k[2 + computes..], ["barrier", "copy", "copy", "barrier"]);
    assert_eq!(kinds(drain), vec!["compute"; computes]);
    assert_eq!((head.unroll, steady.unroll, drain.unroll), (1, 1, 1));
    assert_eq!(count(&prog, |s| matches!(s, Stmt::Sync(Sync::Fence))), 0, "no fence over one slot");
}
