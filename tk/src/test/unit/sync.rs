//! The barrier pass: shared-memory effects outside the schedule templates
//! are fenced, a pipeline's ring only where it meets the code around it.

use test_case::test_case;

use super::attention::probe_spec;
use super::gemm_spec;
use crate::build::*;
use crate::ir::*;
use crate::kernels::Batch;
use crate::kernels::attention::{AttnMask, AttnSpec, CacheLayout, FaCfg, attention};
use crate::kernels::gemm::gemm;
use crate::kernels::nearest::{Floors, Metric, NearestCfg, NearestSpec, nearest};
use crate::lower::sync::insert_barriers;
use crate::ops::config::gemm_cfg;
use crate::sweep::assumed;

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

fn out_view<'k>(k: &mut Kernel<'k>, out: ParamRef<'k, BF16, Out>, at: usize) -> Gmem<'k, BF16, Out> {
    k.view(out, at, [16, 1], Shape::new(16, 16), [None, None])
}

/// Inside a [`Stmt::Warps`] a warp's own store and load of its region are
/// fenced among its lanes alone; what the warps left is fenced for the block
/// before the block reads it.
#[test]
fn warps_fence_their_own_effects_and_the_block_fences_theirs() {
    let mut prog = Kernel::record("warps", |k| {
        k.warps(4);
        let scratch = k.smem::<BF16>("scratch", 4 * 256);
        let out = k.output::<BF16>("o", 2 * 256);
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
        let o = out_view(k, out, 256);
        k.store(o, merged);
    });
    insert_barriers(&mut prog).unwrap();
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
    let spec = assumed(probe_spec(1, 1, 1, 64, 64, AttnMask::default()), cfg);
    let mut prog = attention::<BF16>(&spec);
    insert_barriers(&mut prog).unwrap();
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

/// A loop may run no trip, so what was written before it is still unfenced
/// after it even when its body ends on a barrier.
#[test]
fn a_store_before_a_loop_is_fenced_past_it() {
    let mut prog = Kernel::record("zero_trips", |k| {
        let n = k.var("n", 0, 4);
        let s = k.smem::<F32>("s", 256);
        let tile = k.smem_view::<F32>(s, 0, Shape::new(16, 16));
        let x = k.zeros::<F32>(Shape::new(16, 16));
        k.store(tile, x);
        k.loop_vec::<F32>(n, vec![], |_, _, c| c);
        let y = k.load(tile);
        let out = k.output::<F32>("o", 256);
        let o = k.view(out, 0, [16, 1], Shape::new(16, 16), [None, None]);
        k.store(o, y);
    });
    let Some(Stmt::Loop(l)) = prog.body.0.iter_mut().find(|s| matches!(s, Stmt::Loop(_))) else { panic!("a loop") };
    l.body.0.push(Stmt::Sync(Sync::Barrier { scope: Scope::Block }));
    insert_barriers(&mut prog).unwrap();
    let k = top(&prog);
    let at = k.iter().position(|k| *k == "loop").unwrap();
    assert_eq!(k[at + 1..at + 3], ["barrier", "load"], "{k:?}");
}

/// A 16×16 tile `s` an asynchronous copy fills outside a pipeline.
fn copied<'k>(k: &mut Kernel<'k>) -> SmemRef<'k, BF16> {
    let src = k.input::<BF16>("x", 256);
    let g = k.view(src, 0, [16, 1], Shape::new(16, 16), [None, None]);
    let s = k.smem::<BF16>("s", 256);
    let tile = k.smem_view::<BF16>(s, 0, Shape::new(16, 16));
    k.stage(tile, g, CopyMode::Async);
    s
}

/// Four warps' 16×16 tiles of `scratch`, each filled by its warp's
/// asynchronous copy.
fn copied_by_warps<'k>(k: &mut Kernel<'k>) -> SmemRef<'k, BF16> {
    k.warps(4);
    let src = k.input::<BF16>("x", 4 * 256);
    let scratch = k.smem::<BF16>("scratch", 4 * 256);
    k.each_warp(|k| {
        let w = k.warp();
        let g = k.view(src, w.clone() * 256, [16, 1], Shape::new(16, 16), [None, None]);
        let region = k.smem_view::<BF16>(scratch, w * 256, Shape::new(16, 16));
        k.stage(region, g, CopyMode::Async);
    });
    scratch
}

/// An asynchronous copy is committed, waited for and fenced before its
/// tile is read: one outside a pipeline, and a warp's still in flight when
/// the warps end, landed, not only fenced, before the block reads its tile.
#[test_case(copied, ["async", "commit", "wait", "barrier", "load", "out"]; "outside a pipeline")]
#[test_case(copied_by_warps, ["other", "commit", "wait", "barrier", "load", "out"]; "a warp's past the warps")]
fn an_async_copy_is_waited_for_before_a_read(copy: for<'k> fn(&mut Kernel<'k>) -> SmemRef<'k, BF16>, want: [&str; 6]) {
    let mut prog = Kernel::record("async", |k| {
        let s = copy(k);
        let first = k.smem_view::<BF16>(s, 0, Shape::new(16, 16));
        let y = k.load(first);
        let out = k.output::<BF16>("o", 256);
        let o = out_view(k, out, 0);
        k.store(o, y);
    });
    insert_barriers(&mut prog).unwrap();
    assert_eq!(top(&prog), want);
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
    let mut prog = Kernel::record("async_then_barrier", |k| {
        let src = k.input::<BF16>("x", 256);
        let g = k.view(src, 0, [16, 1], Shape::new(16, 16), [None, None]);
        let a = k.smem::<BF16>("a", 256);
        let ta = k.smem_view::<BF16>(a, 0, Shape::new(16, 16));
        let out = k.output::<BF16>("o", 768);
        k.stage(ta, g, CopyMode::Async);
        match between {
            Between::Fence => {
                let b = k.smem::<BF16>("b", 256);
                let tb = k.smem_view::<BF16>(b, 0, Shape::new(16, 16));
                let z = k.zeros::<BF16>(Shape::new(16, 16));
                k.store(tb, z);
                let yb = k.load(tb);
                let o = out_view(k, out, 512);
                k.store(o, yb);
            }
            Between::Loop => {
                let n = k.var("n", 0, 4);
                k.loop_vec::<F32>(n, vec![], |_, _, c| c);
            }
            Between::Barrier => {}
        }
        for at in [0, 256] {
            let y = k.load(ta);
            let o = out_view(k, out, at);
            k.store(o, y);
        }
    });
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
    insert_barriers(&mut prog).unwrap();
    let k = top(&prog);
    let reads: Vec<usize> = (0..k.len()).filter(|&i| k[i] == "load").collect();
    let [.., first, second] = reads[..] else { panic!("two reads of a: {k:?}") };
    assert_eq!(k[first - 3..first], ["commit", "wait", "barrier"], "{k:?}");
    assert!(!k[first..second].contains(&"wait"), "{k:?}");
}

/// A pipeline over a 16×16 ring in `ring`, summing its steps, with `before`
/// and `after` run around it.
fn around_a_pipeline(
    before: impl for<'k> FnOnce(&mut Kernel<'k>, Ring<'k, BF16>),
    after: impl for<'k> FnOnce(&mut Kernel<'k>, Ring<'k, BF16>),
) -> Program {
    Kernel::record("around_a_pipeline", |k| {
        let x = k.input::<BF16>("x", 4 * 256);
        let g = k.view(x, 0, [16, 1], Shape::new(16, 16), [None, None]);
        let ring = k.ring::<BF16>("ring", 256, 2);
        before(k, ring);
        let init = k.zeros::<F32>(Shape::new(16, 16));
        let [sum] = k.pipeline(
            4,
            ring,
            [init],
            |k, step, slot| {
                let t = k.slot(ring, &slot, Shape::new(16, 16));
                let src = k.at(g, step * 16, 0);
                k.stage(t, src, CopyMode::Async);
            },
            |k, _step, slot, [sum]| {
                let t = k.slot(ring, &slot, Shape::new(16, 16));
                let y = k.load(t);
                let y = k.cast::<BF16, F32>(y);
                [k.binary(*sum, y, BinaryOp::Add).into()]
            },
        );
        after(k, ring);
        let out = k.output::<F32>("o", 256);
        let o = k.view(out, 0, [16, 1], Shape::new(16, 16), [None, None]);
        k.store(o, *sum);
    })
}

fn zero_slot<'k>(k: &mut Kernel<'k>, ring: Ring<'k, BF16>) -> Shared<'k, BF16> {
    // SAFETY: outside the pipeline, which the tests' fences order it against.
    let t = unsafe { k.ring_scratch::<BF16>(ring, 0, Shape::new(16, 16)) };
    let z = k.zeros::<BF16>(Shape::new(16, 16));
    k.store(t, z);
    t
}

/// An async copy filling a tile other than the ring.
fn copy_another<'k>(k: &mut Kernel<'k>, _: Ring<'k, BF16>) {
    let x = k.input::<BF16>("y", 256);
    let g = k.view(x, 0, [16, 1], Shape::new(16, 16), [None, None]);
    let other = k.smem::<BF16>("other", 256);
    let t = k.smem_view::<BF16>(other, 0, Shape::new(16, 16));
    k.stage(t, g, CopyMode::Async);
}

/// Code around a pipeline is fenced against its ring where they meet: the
/// template's first copies overwrite the ring, so a read of it before the
/// pipeline is fenced at entry; no barrier follows the last consume, so a
/// write after it is fenced; and an async copy still filling another tile
/// is landed before the pipeline, not left to the template's group count.
#[test_case(|k, ring| { let t = zero_slot(k, ring); let _ = k.load(t); }, |_, _| {},
    &["other", "store", "barrier", "load", "other", "barrier", "pipeline", "out"]; "a ring read before its pipeline")]
#[test_case(|_, _| {}, |k, ring| { zero_slot(k, ring); },
    &["other", "pipeline", "other", "barrier", "store", "out"]; "a ring written after its pipeline")]
#[test_case(copy_another, |_, _| {}, &["async", "other", "commit", "wait", "barrier", "pipeline", "out"]; "an async copy before a pipeline")]
fn a_pipeline_is_fenced_where_the_code_around_it_meets_it(
    before: for<'k> fn(&mut Kernel<'k>, Ring<'k, BF16>),
    after: for<'k> fn(&mut Kernel<'k>, Ring<'k, BF16>),
    want: &[&str],
) {
    let mut prog = around_a_pipeline(before, after);
    insert_barriers(&mut prog).unwrap();
    assert_eq!(top(&prog), want);
}

/// A consume body starts after the template's barrier: only what it does to
/// its own tiles within one step is fenced.
#[test]
fn a_consume_body_fences_its_own_tiles() {
    let mut prog = Kernel::record("consume_scratch", |k| {
        let x = k.input::<BF16>("x", 4 * 256);
        let g = k.view(x, 0, [16, 1], Shape::new(16, 16), [None, None]);
        let ring = k.ring::<BF16>("ring", 256, 2);
        let scratch = k.smem::<BF16>("scratch", 256);
        let init = k.zeros::<BF16>(Shape::new(16, 16));
        let [last] = k.pipeline(
            4,
            ring,
            [init],
            |k, step, slot| {
                let t = k.slot(ring, &slot, Shape::new(16, 16));
                let src = k.at(g, step * 16, 0);
                k.stage(t, src, CopyMode::Async);
            },
            |k, _step, slot, _| {
                let t = k.slot(ring, &slot, Shape::new(16, 16));
                let y = k.load(t);
                let s = k.smem_view::<BF16>(scratch, 0, Shape::new(16, 16));
                k.store(s, y);
                [k.load(s).into()]
            },
        );
        let out = k.output::<BF16>("o", 256);
        let o = out_view(k, out, 0);
        k.store(o, *last);
    });
    insert_barriers(&mut prog).unwrap();
    let Some(Stmt::Pipeline(p)) = prog.body.0.iter().find(|s| matches!(s, Stmt::Pipeline(_))) else {
        panic!("a pipeline")
    };
    assert_eq!(kinds(&prog, &p.consume.body), ["load", "store", "barrier", "load"]);
}

const BOTH: Floors = Floors { row: true, col: true };

/// 100 bf16 rows against 150 candidates at depth `d` under L2, grouped,
/// floored by `floors`, keeping `k` scores a row.
fn nearest_spec(d: usize, floors: Floors, k: usize) -> NearestSpec {
    let (dtype, metric) = (svod_dtype::ScalarDType::BFloat16, Metric::L2);
    NearestSpec { n: 100, m: 150, d, dtype, metric, floors, groups: true, k, live: Default::default() }
}

/// The shipped kernels keep their shared memory in pipeline rings, which
/// their templates fence: the pass leaves them as they are.
#[test_case(gemm::<BF16>(&assumed(gemm_spec(256, 256, 256), gemm_cfg([128, 128, 32], 3, [2, 2], true))); "gemm")]
#[test_case(attention::<BF16>(&assumed(
    AttnSpec {
        batch: Batch::var("b", 1, 2),
        t: 4, tk: 200, heads: 4, kv_heads: 2, d: 64,
        mask: AttnMask { causal: true, key_lens: true, key_mask: true, ..AttnMask::default() },
        cache: Some(CacheLayout { rows: 3, heads_total: 6, row_map: true, appended: 4, app_lens: true }),
        packed: false,
        scale: 0.125,
    },
    FaCfg { splits: 2, ..FaCfg::new(16, 64, 3) },
)); "attention with every mask, a cache and splits")]
#[test_case(nearest::<BF16>(&assumed(nearest_spec(32, BOTH, 1), NearestCfg::new(32, 32, 3))); "nearest with floors and groups")]
#[test_case(nearest::<BF16>(&assumed(nearest_spec(32, BOTH, 8), NearestCfg::new(32, 32, 3))); "nearest lists with floors and groups")]
fn shipped_kernels_need_no_barriers(prog: Program) {
    let mut fenced = prog.clone();
    insert_barriers(&mut fenced).unwrap();
    assert!(fenced == prog);
}

/// A tile stored and read back under a branch needs a block barrier between
/// the two: under a branch on the block every warp meets it, under one on
/// the warp some skip it and the rest would wait forever, so the lowering
/// refuses it; inside the warps' own statement the barrier is the warp's.
#[test_case(false, false => Ok(vec!["other", "store", "barrier", "load", "out"]); "a branch on the block")]
#[test_case(true, false => Err("a shared tile's fence is a block barrier under a branch on the warp, which some warps skip".into()); "a branch on the warp")]
#[test_case(true, true => Ok(vec!["other", "store", "barrier", "load", "out"]); "a branch on the warp inside the warps")]
fn a_block_barrier_under_a_branch_on_the_warp_is_refused(
    on_warp: bool,
    own: bool,
) -> Result<Vec<&'static str>, String> {
    fn body<'k>(k: &mut Kernel<'k>, s: SmemRef<'k, F32>, out: ParamRef<'k, F32, Out>) {
        let tile = k.smem_view::<F32>(s, 0, Shape::new(16, 16));
        let x = k.zeros::<F32>(Shape::new(16, 16));
        k.store(tile, x);
        let y = k.load(tile);
        let o = k.view(out, 0, [16, 1], Shape::new(16, 16), [None, None]);
        k.store(o, y);
    }
    let mut prog = Kernel::record("divergent", |k| {
        let s = k.smem::<F32>("s", 256);
        let out = k.output::<F32>("o", 256);
        let pred = if on_warp { k.warp().lt(2) } else { k.block(0).lt(2) };
        if own {
            k.each_warp(|k| k.if_(pred, |k| body(k, s, out), |_| {}));
        } else {
            k.if_(pred, |k| body(k, s, out), |_| {});
        }
    });
    insert_barriers(&mut prog).map_err(|e| e.to_string())?;
    let mut stmt = &prog.body.0[0];
    if let Stmt::Warps { body } = stmt {
        stmt = &body.0[0];
    }
    let Stmt::If { then, .. } = stmt else { panic!("a branch") };
    Ok(kinds(&prog, then))
}

/// A branch on a vote is a branch on the warp: a shared tile stored and read
/// back under it would need a block barrier some warps skip, so the lowering
/// refuses it; reductions within the warp need none.
#[test_case(true => Err("a shared tile's fence is a block barrier under a branch on the warp, which some warps skip".into()); "through shared memory")]
#[test_case(false => Ok(()); "in registers")]
fn a_block_barrier_under_a_vote_is_refused(shared: bool) -> Result<(), String> {
    let mut prog = Kernel::record("voted", |k| {
        let s = k.smem::<F32>("s", 256);
        let x = k.zeros::<F32>(Shape::new(16, 16));
        let low = k.reduce(x, Axis::Row, ReduceOp::Min);
        let zero = k.zeros::<F32>(Shape::new(16, 1));
        let pred = k.compare(low, zero, CmpOp::Lt);
        k.where_any(
            pred,
            |k| {
                if shared {
                    let tile = k.smem_view::<F32>(s, 0, Shape::new(16, 16));
                    k.store(tile, x);
                    let y = k.load(tile);
                    return [k.reduce(y, Axis::Row, ReduceOp::Max)];
                }
                [k.reduce(x, Axis::Row, ReduceOp::Max)]
            },
            [low],
        );
    });
    insert_barriers(&mut prog).map_err(|e| e.to_string())
}

/// A barrier the kernel places itself is held to the same rule: under a vote
/// some warps never reach it.
#[test_case(true => Err("a barrier is a block barrier under a branch on the warp, which some warps skip".into()); "under a vote")]
#[test_case(false => Ok(()); "before it")]
fn an_explicit_barrier_under_a_vote_is_refused(voted: bool) -> Result<(), String> {
    let mut prog = Kernel::record("voted", |k| {
        let x = k.zeros::<F32>(Shape::new(16, 16));
        let low = k.reduce(x, Axis::Row, ReduceOp::Min);
        let zero = k.zeros::<F32>(Shape::new(16, 1));
        let pred = k.compare(low, zero, CmpOp::Lt);
        if !voted {
            k.barrier();
        }
        k.if_any(
            pred,
            |k| {
                if voted {
                    k.barrier();
                }
                vec![k.reduce(x, Axis::Row, ReduceOp::Max)]
            },
            vec![low],
        );
    });
    insert_barriers(&mut prog).map_err(|e| e.to_string())
}

/// A streamed nearest runs the depth pipeline once per candidate tile: the
/// pass fences the ring where the next tile's first fill meets the last
/// tile's reads of it, one barrier at the head of the pipeline in the loop,
/// and adds nothing else.
#[test]
fn a_ring_restarted_in_a_loop_is_fenced_at_its_head() {
    let spec = assumed(nearest_spec(80, Floors::default(), 1), NearestCfg::streamed(32, 32, 32, 3));
    let prog = nearest::<BF16>(&spec);
    let mut fenced = prog.clone();
    insert_barriers(&mut fenced).unwrap();
    let body = |prog: &Program| {
        let Some(Stmt::Loop(l)) = prog.body.0.iter().find(|s| matches!(s, Stmt::Loop(_))) else { panic!("a loop") };
        l.body.clone()
    };
    let (before, after) = (body(&prog), body(&fenced));
    let at = before.0.iter().position(|s| matches!(s, Stmt::Pipeline(_))).expect("a pipeline");
    assert_eq!(kinds(&fenced, &after)[at..at + 2], ["barrier", "pipeline"]);
    let mut unfenced = after.0.clone();
    unfenced.remove(at);
    assert!(unfenced == before.0);
    assert!(top(&fenced) == top(&prog));
}
