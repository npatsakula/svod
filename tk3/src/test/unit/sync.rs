//! The barrier pass: shared-memory effects outside the schedule templates
//! are fenced, a pipeline's ring only where it meets the code around it.

use test_case::test_case;

use crate::build::*;
use crate::ir::*;
use crate::kernels::Batch;
use crate::kernels::attention::{AttnMask, AttnSpec, Cache, FaCfg, attention};
use crate::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use crate::lower::sync::insert_barriers;

/// The statements of `block` by kind.
fn kinds(prog: &Program, block: &Block) -> Vec<&'static str> {
    block
        .0
        .iter()
        .map(|s| match s {
            Stmt::Copy { mode: CopyMode::Async, .. } => "async",
            Stmt::Copy { dst, .. } if prog.value(*dst).tier() == Tier::Smem => "store",
            Stmt::Copy { dst, .. } if prog.value(*dst).tier() == Tier::Global => "out",
            Stmt::Copy { .. } => "load",
            Stmt::Sync(Sync::Barrier { .. }) => "barrier",
            Stmt::Sync(Sync::CommitAsync) => "commit",
            Stmt::Sync(Sync::WaitAsync { .. }) => "wait",
            Stmt::Loop(_) => "loop",
            Stmt::Pipeline(_) => "pipeline",
            _ => "other",
        })
        .collect()
}

fn top(prog: &Program) -> Vec<&'static str> {
    kinds(prog, &prog.body)
}

fn out_view(k: &mut Kernel, out: ParamRef<BF16>, at: usize) -> Gmem<BF16> {
    k.view(out, at, [16, 1], Shape::new(16, 16), [None, None])
}

/// Inside a [`Stmt::Warps`] a warp's own store and load of its region are
/// fenced among its lanes alone; what the warps left is fenced for the block
/// before the block reads it.
#[test]
fn warps_fence_their_own_effects_and_the_block_fences_theirs() {
    let mut k = Kernel::new("warps");
    k.warps(4);
    let scratch = k.smem::<BF16>("scratch", 4 * 256);
    let out = k.param::<BF16>("o", ParamKind::Out, 2 * 256);
    k.each_warp(|k| {
        let w = k.warp();
        let region = k.smem_view::<BF16>(scratch, w * 256, Shape::new(16, 16));
        let z = k.zeros::<BF16>(Shape::new(16, 16));
        k.store(region, z);
        let back = k.load(region);
        let o = out_view(k, out, 0);
        k.store(o, back);
    });
    let first = k.smem_view::<BF16>(scratch, 0, Shape::new(16, 16));
    let merged = k.load(first);
    let o = out_view(&mut k, out, 256);
    k.store(o, merged);
    let mut prog = k.finish();
    insert_barriers(&mut prog);
    assert_eq!(
        top(&prog),
        ["other", "barrier", "load", "out"],
        "the block fences the warps' effects: {:?}",
        top(&prog)
    );
    let Stmt::Warps { body } = &prog.body.0[0] else { panic!("the warps statement first") };
    assert_eq!(kinds(&prog, body), ["other", "store", "barrier", "load", "out"]);
    let scopes: Vec<Scope> = prog
        .walk()
        .filter_map(|(_, s)| if let Stmt::Sync(Sync::Barrier { scope }) = s { Some(*scope) } else { None })
        .collect();
    assert_eq!(scopes, [Scope::Warp, Scope::Block]);
}

/// A key warp's merged state goes over its own ring slots once its
/// pipeline ends: its output store, an f32 view of the keys it read, is
/// fenced among its lanes alone, and the block fences the warps' states
/// before it folds them.
#[test]
fn a_key_warps_state_over_its_ring_is_fenced_by_the_warp_then_the_block() {
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
    insert_barriers(&mut prog);
    assert!(
        prog.smem.iter().all(|a| a.dtype == svod_dtype::ScalarDType::BFloat16),
        "no merge allocation: {:?}",
        prog.smem
    );
    let f32_smem = |prog: &Program, v: ValId| {
        let val = prog.value(v);
        val.tier() == Tier::Smem && val.dtype == svod_dtype::ScalarDType::Float32
    };
    let at = prog.body.0.iter().position(|s| matches!(s, Stmt::Warps { .. })).expect("the warps statement");
    let Stmt::Warps { body } = &prog.body.0[at] else { unreachable!() };
    let store = body
        .0
        .iter()
        .position(|s| matches!(s, Stmt::Copy { dst, .. } if f32_smem(&prog, *dst)))
        .expect("the output store");
    assert!(matches!(body.0[store - 1], Stmt::Sync(Sync::Barrier { scope: Scope::Warp })), "{:?}", kinds(&prog, body));
    let after = &prog.body.0[at + 1..];
    let load =
        after.iter().position(|s| matches!(s, Stmt::Copy { src, .. } if f32_smem(&prog, *src))).expect("a merge load");
    assert!(
        after[..load].iter().any(|s| matches!(s, Stmt::Sync(Sync::Barrier { scope: Scope::Block }))),
        "{:?}",
        top(&prog)
    );
}

/// A warp's asynchronous copy still in flight when the warps end is landed,
/// not only fenced, before the block reads its tile.
#[test]
fn a_warps_async_copy_is_landed_before_the_block_reads_it() {
    let mut k = Kernel::new("warps_async");
    k.warps(4);
    let src = k.param::<BF16>("x", ParamKind::In, 4 * 256);
    let scratch = k.smem::<BF16>("scratch", 4 * 256);
    k.each_warp(|k| {
        let w = k.warp();
        let g = k.view(src, w.clone() * 256, [16, 1], Shape::new(16, 16), [None, None]);
        let region = k.smem_view::<BF16>(scratch, w * 256, Shape::new(16, 16));
        k.stage(region, g, CopyMode::Async);
    });
    let first = k.smem_view::<BF16>(scratch, 0, Shape::new(16, 16));
    let merged = k.load(first);
    let out = k.param::<BF16>("o", ParamKind::Out, 256);
    let o = out_view(&mut k, out, 0);
    k.store(o, merged);
    let mut prog = k.finish();
    insert_barriers(&mut prog);
    assert_eq!(top(&prog), ["other", "commit", "wait", "barrier", "load", "out"]);
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
    l.body.0.push(Stmt::Sync(Sync::Barrier { scope: Scope::Block }));
    insert_barriers(&mut prog);
    let k = top(&prog);
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
    let o = out_view(&mut k, out, 0);
    k.store(o, y);
    let mut prog = k.finish();
    insert_barriers(&mut prog);
    assert_eq!(top(&prog), ["async", "commit", "wait", "barrier", "load", "out"]);
}

/// Where a barrier meets a tile an async copy is filling: the pass's own
/// fence for another tile, an author's barrier, or a loop whose body ends on
/// one.
#[derive(Clone, Copy)]
enum Between {
    Fence,
    Barrier,
    Loop,
}

/// An async copy into `a`, then `between`, then two reads of `a`: a barrier
/// orders threads but lands no copy, so the first read still waits for it,
/// and the second, after that wait, does not wait again.
#[test_case(Between::Fence; "the fence for another tile")]
#[test_case(Between::Barrier; "an author's barrier")]
#[test_case(Between::Loop; "a loop ending on a barrier")]
fn a_pending_async_copy_survives_a_barrier(between: Between) {
    let mut k = Kernel::new("async_then_barrier");
    let src = k.param::<BF16>("x", ParamKind::In, 256);
    let g = k.view(src, 0, [16, 1], Shape::new(16, 16), [None, None]);
    let a = k.smem::<BF16>("a", 256);
    let ta = k.smem_view::<BF16>(a, 0, Shape::new(16, 16));
    let out = k.param::<BF16>("o", ParamKind::Out, 768);
    k.stage(ta, g, CopyMode::Async);
    match between {
        Between::Fence => {
            let b = k.smem::<BF16>("b", 256);
            let tb = k.smem_view::<BF16>(b, 0, Shape::new(16, 16));
            let z = k.zeros::<BF16>(Shape::new(16, 16));
            k.store(tb, z);
            let yb = k.load(tb);
            let o = out_view(&mut k, out, 512);
            k.store(o, yb);
        }
        Between::Loop => {
            let n = k.var("n", 0, 4);
            k.loop_::<F32, 0>(n, [], |_, _, c| c);
        }
        Between::Barrier => {}
    }
    for at in [0, 256] {
        let y = k.load(ta);
        let o = out_view(&mut k, out, at);
        k.store(o, y);
    }
    let mut prog = k.finish();
    let after_copy = top(&prog).iter().position(|s| *s == "async").unwrap() + 1;
    match between {
        Between::Barrier => prog.body.0.insert(after_copy, Stmt::Sync(Sync::Barrier { scope: Scope::Block })),
        Between::Loop => {
            let Some(Stmt::Loop(l)) = prog.body.0.iter_mut().find(|s| matches!(s, Stmt::Loop(_))) else {
                panic!("a loop")
            };
            l.body.0.push(Stmt::Sync(Sync::Barrier { scope: Scope::Block }));
        }
        Between::Fence => {}
    }
    insert_barriers(&mut prog);
    let k = top(&prog);
    let reads: Vec<usize> = (0..k.len()).filter(|&i| k[i] == "load").collect();
    let [.., first, second] = reads[..] else { panic!("two reads of a: {k:?}") };
    assert_eq!(k[first - 3..first], ["commit", "wait", "barrier"], "{k:?}");
    assert!(!k[first..second].contains(&"wait"), "{k:?}");
}

/// A pipeline over a 16×16 ring in `ring`, summing its steps, with `before`
/// and `after` run around it.
fn around_a_pipeline(before: impl FnOnce(&mut Kernel, SmemId), after: impl FnOnce(&mut Kernel, SmemId)) -> Program {
    let mut k = Kernel::new("around_a_pipeline");
    let x = k.param::<BF16>("x", ParamKind::In, 4 * 256);
    let g = k.view(x, 0, [16, 1], Shape::new(16, 16), [None, None]);
    let ring = k.smem::<BF16>("ring", 2 * 256);
    before(&mut k, ring);
    let init = k.zeros::<F32>(Shape::new(16, 16));
    let [sum] = k.pipeline(
        4,
        2,
        [init],
        |k, step, slot| {
            let t = k.smem_slot::<BF16>(ring, slot, Shape::new(16, 16));
            let src = k.at(g, step * 16, 0);
            k.stage(t, src, CopyMode::Async);
        },
        |k, _step, slot, [sum]| {
            let t = k.smem_slot::<BF16>(ring, slot, Shape::new(16, 16));
            let y = k.load(t);
            let y = k.cast::<BF16, F32>(y);
            [k.binary(sum, y, BinaryOp::Add)]
        },
    );
    after(&mut k, ring);
    let out = k.param::<F32>("o", ParamKind::Out, 256);
    let o = k.view(out, 0, [16, 1], Shape::new(16, 16), [None, None]);
    k.store(o, sum);
    k.finish()
}

fn zero_slot(k: &mut Kernel, ring: SmemId) -> Shared<BF16> {
    let t = k.smem_view::<BF16>(ring, 0, Shape::new(16, 16));
    let z = k.zeros::<BF16>(Shape::new(16, 16));
    k.store(t, z);
    t
}

/// The template's first copies overwrite the ring, so a read of it before
/// the pipeline is fenced first.
#[test]
fn a_ring_read_before_its_pipeline_is_fenced_at_entry() {
    let mut prog = around_a_pipeline(
        |k, ring| {
            let t = zero_slot(k, ring);
            let _ = k.load(t);
        },
        |_, _| {},
    );
    insert_barriers(&mut prog);
    let k = top(&prog);
    let at = k.iter().position(|s| *s == "pipeline").unwrap();
    assert_eq!(k[..at], ["other", "store", "barrier", "load", "other", "barrier"]);
}

/// No barrier follows the last consume, so its reads of the ring are open
/// after the pipeline.
#[test]
fn a_ring_written_after_its_pipeline_is_fenced() {
    let mut prog = around_a_pipeline(
        |_, _| {},
        |k, ring| {
            zero_slot(k, ring);
        },
    );
    insert_barriers(&mut prog);
    let k = top(&prog);
    let at = k.iter().position(|s| *s == "pipeline").unwrap();
    assert_eq!(k[at + 1..at + 4], ["other", "barrier", "store"], "{k:?}");
}

/// An async copy still filling another tile is landed before a pipeline,
/// not left to the template's group count.
#[test]
fn an_async_copy_is_landed_before_a_pipeline() {
    let mut prog = around_a_pipeline(
        |k, _| {
            let x = k.param::<BF16>("y", ParamKind::In, 256);
            let g = k.view(x, 0, [16, 1], Shape::new(16, 16), [None, None]);
            let other = k.smem::<BF16>("other", 256);
            let t = k.smem_view::<BF16>(other, 0, Shape::new(16, 16));
            k.stage(t, g, CopyMode::Async);
        },
        |_, _| {},
    );
    insert_barriers(&mut prog);
    let k = top(&prog);
    let at = k.iter().position(|s| *s == "pipeline").unwrap();
    assert_eq!(k[at - 3..at], ["commit", "wait", "barrier"], "{k:?}");
}

/// A consume body starts after the template's barrier: only what it does to
/// its own tiles within one step is fenced.
#[test]
fn a_consume_body_fences_its_own_tiles() {
    let mut k = Kernel::new("consume_scratch");
    let x = k.param::<BF16>("x", ParamKind::In, 4 * 256);
    let g = k.view(x, 0, [16, 1], Shape::new(16, 16), [None, None]);
    let ring = k.smem::<BF16>("ring", 2 * 256);
    let scratch = k.smem::<BF16>("scratch", 256);
    let init = k.zeros::<BF16>(Shape::new(16, 16));
    let [last] = k.pipeline(
        4,
        2,
        [init],
        |k, step, slot| {
            let t = k.smem_slot::<BF16>(ring, slot, Shape::new(16, 16));
            let src = k.at(g, step * 16, 0);
            k.stage(t, src, CopyMode::Async);
        },
        |k, _step, slot, _| {
            let t = k.smem_slot::<BF16>(ring, slot, Shape::new(16, 16));
            let y = k.load(t);
            let s = k.smem_view::<BF16>(scratch, 0, Shape::new(16, 16));
            k.store(s, y);
            [k.load(s)]
        },
    );
    let out = k.param::<BF16>("o", ParamKind::Out, 256);
    let o = out_view(&mut k, out, 0);
    k.store(o, last);
    let mut prog = k.finish();
    insert_barriers(&mut prog);
    let Some(Stmt::Pipeline(p)) = prog.body.0.iter().find(|s| matches!(s, Stmt::Pipeline(_))) else {
        panic!("a pipeline")
    };
    assert_eq!(kinds(&prog, &p.consume.body), ["load", "store", "barrier", "load"]);
}

/// The shipped kernels keep their shared memory in pipeline rings, which
/// their templates fence: the pass leaves them as they are.
#[test_case(gemm::<BF16>(&GemmSpec {
    m: 256, n: 256, k: 256, batch: Batch::Static(1), epilogue: Epilogue::default(),
    cfg: GemmCfg { tile: [128, 128, 32], stages: 3, warps: [2, 2], group_m: 8, unroll: true },
}); "gemm")]
#[test_case(attention::<BF16>(&AttnSpec {
    batch: Batch::Var { name: "b".into(), min: 1, max: 2 },
    t: 4, tk: 200, heads: 4, kv_heads: 2, d: 64,
    mask: AttnMask { causal: true, key_lens: true, key_mask: true, ..AttnMask::default() },
    cache: Some(Cache { rows: 3, heads_total: 6, row_map: true, appended: 4, app_lens: true }),
    scale: 0.125,
    cfg: FaCfg { splits: 2, ..FaCfg::new(16, 64, 3) },
}); "attention with every mask, a cache and splits")]
fn shipped_kernels_need_no_barriers(prog: Program) {
    let mut fenced = prog.clone();
    insert_barriers(&mut fenced);
    assert!(fenced == prog);
}
