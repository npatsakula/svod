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
/// the guarded prologue/drain) and leaves no pipeline behind.
#[test_case(Prefetch::CpAsync, 1; "cp.async one stage")]
#[test_case(Prefetch::CpAsync, 2; "cp.async two stages")]
#[test_case(Prefetch::CpAsync, 4; "cp.async four stages")]
#[test_case(Prefetch::RegisterStaged, 2; "register staged")]
fn expansion_preserves_the_result(prefetch: Prefetch, stages: usize) {
    let (m, n, k) = (64usize, 32usize, 160usize);
    let mut prog = super::programs::gemm_nt(m, n, k, 32, 32, 32, stages);
    let a: Vec<f64> = (0..m * k).map(|i| round_to(ScalarDType::BFloat16, ((i * 13) % 17) as f64 / 8.0 - 1.0)).collect();
    let b: Vec<f64> = (0..n * k).map(|i| round_to(ScalarDType::BFloat16, ((i * 7) % 19) as f64 / 8.0 - 1.0)).collect();
    let want = run(&prog, vec![a.clone(), b.clone(), vec![0.0; m * n]], &[("b", 1)]).unwrap();
    expand(&mut prog, Schedule::Uniform { prefetch, unroll: true });
    let got = run(&prog, vec![a, b, vec![0.0; m * n]], &[("b", 1)]).unwrap();
    assert_eq!(got[2], want[2]);
    assert_eq!(count(&prog, |s| matches!(s, Stmt::Pipeline(_))), 0);
    assert_eq!(count(&prog, |s| matches!(s, Stmt::Loop(_))), 1);
    let barriers = count(&prog, |s| matches!(s, Stmt::Sync(Sync::Barrier { .. })));
    assert_eq!(barriers, if prefetch == Prefetch::CpAsync && stages == 1 { 2 } else { 1 }, "barriers per iteration");
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

/// Register staging splits every global→shared copy into a load before the
/// consumer and a store after it.
#[test]
fn register_staging_splits_the_copies_around_the_compute() {
    let mut prog = super::programs::gemm_nt(32, 32, 64, 32, 32, 32, 2);
    expand(&mut prog, Schedule::Uniform { prefetch: Prefetch::RegisterStaged, unroll: true });
    let Stmt::Loop(l) = &prog.body.0[1] else { panic!("the pipeline became a loop") };
    let blocks: Vec<&Block> = l
        .body
        .0
        .iter()
        .filter_map(|s| match s {
            Stmt::If { then, .. } => Some(then),
            _ => None,
        })
        .collect();
    let tier = |s: &Stmt| match s {
        Stmt::Copy { dst, src, mode: CopyMode::Staged } => Some((prog.value(*src).tier(), prog.value(*dst).tier())),
        _ => None,
    };
    assert!(blocks[0].0.iter().all(|s| tier(s) == Some((Tier::Global, Tier::Reg))), "issue: global → registers");
    assert!(blocks[2].0.iter().all(|s| tier(s) == Some((Tier::Reg, Tier::Smem))), "commit: registers → shared");
    assert!(matches!(l.body.0.last(), Some(Stmt::Sync(Sync::Barrier { .. }))));
}
