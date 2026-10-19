//! The barrier pass: what an author writes outside the schedule templates
//! is fenced from its effects.

use crate::build::*;
use crate::ir::*;
use crate::lower::sync::insert_barriers;

/// The top-level statements of `prog` by kind.
fn kinds(prog: &Program) -> Vec<&'static str> {
    prog.body
        .0
        .iter()
        .map(|s| match s {
            Stmt::Copy { mode: CopyMode::Async, .. } => "async",
            Stmt::Copy { dst, .. } if prog.value(*dst).tier() == Tier::Smem => "store",
            Stmt::Copy { .. } => "load",
            Stmt::Sync(Sync::Barrier { .. }) => "barrier",
            Stmt::Sync(Sync::CommitAsync) => "commit",
            Stmt::Sync(Sync::WaitAsync { .. }) => "wait",
            Stmt::Loop(_) => "loop",
            _ => "other",
        })
        .collect()
}

/// A loop may run no trip, so what was written before it is still unfenced
/// after it even when its body ends on a barrier.
#[test]
fn a_store_before_a_loop_is_fenced_past_it() {
    let mut k = Kernel::new("zero_trips");
    let n = k.var("n", 0, 4);
    let s = k.smem::<F32>("s", 256);
    let tile = k.smem_view::<F32>(s, 0, Shape::new(16, 16));
    let x = k.zeros::<F32>(Shape::new(16, 16));
    k.store(tile, x);
    k.loop_::<F32, 0>(n, [], |_, _, c| c);
    let y = k.load(tile);
    let out = k.param::<F32>("o", ParamKind::Out, 256);
    let o = k.view(out, 0, [16, 1], Shape::new(16, 16), [None, None]);
    k.store(o, y);
    let mut prog = k.finish();
    let Some(Stmt::Loop(l)) = prog.body.0.iter_mut().find(|s| matches!(s, Stmt::Loop(_))) else { panic!("a loop") };
    l.body.0.push(Stmt::Sync(Sync::Barrier { role: None }));
    insert_barriers(&mut prog);
    let k = kinds(&prog);
    let at = k.iter().position(|k| *k == "loop").unwrap();
    assert_eq!(k[at + 1..at + 3], ["barrier", "load"], "{k:?}");
}

/// An asynchronous copy outside a pipeline is committed, waited for and
/// fenced before its tile is read.
#[test]
fn an_async_copy_is_waited_for_before_a_read() {
    let mut k = Kernel::new("async");
    let src = k.param::<BF16>("x", ParamKind::In, 256);
    let g = k.view(src, 0, [16, 1], Shape::new(16, 16), [None, None]);
    let s = k.smem::<BF16>("s", 256);
    let tile = k.smem_view::<BF16>(s, 0, Shape::new(16, 16));
    k.stage(tile, g, CopyMode::Async);
    let y = k.load(tile);
    let out = k.param::<BF16>("o", ParamKind::Out, 256);
    let o = k.view(out, 0, [16, 1], Shape::new(16, 16), [None, None]);
    k.store(o, y);
    let mut prog = k.finish();
    insert_barriers(&mut prog);
    assert_eq!(kinds(&prog)[..5], ["async", "commit", "wait", "barrier", "load"], "{:?}", kinds(&prog));
}
