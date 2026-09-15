//! White-box tests over `crate::optimizer::implicit_barriers`: RAW/WAR barrier
//! inference and the buffer-aliasing walk that decides which accesses meet.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::{AxisType, Op, UOp, ops};
use test_case::test_case;

use crate::optimizer::implicit_barriers::add_implicit_barriers;
use crate::test::support::prelude::*;

/// A rank-1 buffer in `addrspace`; only global storage carries a device.
fn buffer(slot: usize, addrspace: AddrSpace) -> Arc<UOp> {
    let device = (addrspace == AddrSpace::Global).then_some(DeviceSpec::Cpu);
    UOp::buffer(slot, 8, DType::Float32, addrspace, device)
}

/// `INDEX(wrapper(memory), offset)` — built directly so a non-buffer wrapper can
/// stand in for the buffer the aliasing walk must see through.
fn wrapped_index(kind: usize, memory: Arc<UOp>, offset: Arc<UOp>) -> Arc<UOp> {
    let wrapped = match kind {
        0 => memory,
        1 => memory.cast(DType::Float16),
        2 => UOp::new(Op::Reshape(ops::Reshape { src: memory, new_shape: UOp::index_const(1) }), DType::Float32),
        3 => memory.after(smallvec![]),
        4 => memory.mselect(0),
        5 => UOp::new(Op::BitCast(ops::BitCast { src: memory, dtype: DType::Float16 }), DType::Float16),
        6 => UOp::mstack(smallvec![memory.clone(), memory]),
        other => panic!("unknown wrapper {other}"),
    };
    UOp::new(Op::Index(ops::Index { buffer: wrapped, indices: smallvec![offset] }), DType::Float32)
}

// RAW BARRIERS (AFTER)

/// A LOCAL `AFTER` whose dependency chain holds an unbarriered store gets a
/// BARRIER around that store; global memory is not thread-shared.
#[test_case(AddrSpace::Local, true; "local memory is barriered")]
#[test_case(AddrSpace::Global, false; "global memory is not thread-shared")]
fn after_store_barrier_follows_the_addrspace(addrspace: AddrSpace, barriered: bool) {
    let memory = buffer(0, addrspace);
    let stored = index_of(memory.clone(), index_const(0)).store(UOp::native_const(1.0f32));
    let result = add_implicit_barriers(memory.clone().after(smallvec![stored.clone()]));

    let (passthrough, deps) = expect_after(&result);
    assert_same!(passthrough, memory);
    if barriered {
        assert!(matches!(deps.as_slice(), [barrier]
            if matches!(barrier.op(), Op::Barrier(ops::Barrier { src, deps })
                if Arc::ptr_eq(src, &stored) && deps.is_empty())));
    } else {
        assert!(matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, &stored)));
    }
}

/// A LOCAL store of `value` at `offset`, optionally already barriered.
fn local_store(offset: i64, value: f32, barriered: bool) -> Arc<UOp> {
    let memory = buffer(0, AddrSpace::Local);
    let stored = index_of(memory, index_const(offset)).store(UOp::native_const(value));
    if barriered { stored.barrier(smallvec![]) } else { stored }
}

/// `barrier_from_sources` takes the first dependency as the BARRIER source and
/// wraps the rest, whether or not one of them already carries a BARRIER.
#[test_case(false; "an unbarriered list")]
#[test_case(true; "a partly barriered list")]
fn raw_barrier_keeps_every_dependency(barriered: bool) {
    let memory = buffer(0, AddrSpace::Local);
    let first = index_of(memory.clone(), index_const(0)).store(UOp::native_const(1.0f32));
    let second = local_store(1, 2.0, barriered);
    let result = add_implicit_barriers(memory.after(smallvec![first.clone(), second.clone()]));

    let (_, deps) = expect_after(&result);
    assert!(matches!(deps.as_slice(), [barrier]
        if matches!(barrier.op(), Op::Barrier(ops::Barrier { src, deps })
            if Arc::ptr_eq(src, &first) && matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, &second)))));
}

/// A dependency list that is already fully barriered is left alone.
#[test]
fn existing_barrier_is_not_reinferred() {
    let explicit = local_store(0, 1.0, true);
    let result = add_implicit_barriers(buffer(0, AddrSpace::Local).after(smallvec![explicit.clone()]));

    assert!(matches!(result.op(), Op::After(ops::After { deps, .. })
        if matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, &explicit))));
}

// WAR BARRIERS (END)

/// A LOCAL buffer written and read across at least two iterations of a loop gets
/// a WAR BARRIER whose source is the store and whose dependency is the load.
#[test_case(AxisType::Reduce; "reduce loop")]
#[test_case(AxisType::Weak; "weak loop")]
#[test_case(AxisType::Loop; "plain loop")]
fn local_store_and_load_get_a_war_barrier(axis_type: AxisType) {
    let memory = buffer(0, AddrSpace::Local);
    let range = range(4, axis_type, 0);
    let loaded = load(index_of(memory.clone(), range.clone()));
    let stored = index_of(memory, range.clone()).store(loaded.clone());
    let result = add_implicit_barriers(stored.end(smallvec![range.clone()]));

    let (computation, ranges) = expect_end(&result);
    assert!(matches!(computation.op(), Op::Barrier(ops::Barrier { src, deps })
        if Arc::ptr_eq(src, &stored) && matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, &loaded))));
    assert!(matches!(ranges.as_slice(), [closed] if Arc::ptr_eq(closed, &range)));
}

/// An END whose computation is itself a load participates in WAR detection.
#[test]
fn end_computation_load_participates_in_war_detection() {
    let memory = buffer(0, AddrSpace::Local);
    let range = range(4, AxisType::Weak, 0);
    let stored = index_of(memory.clone(), range.clone()).store(UOp::native_const(1.0f32));
    let loaded = load(index_of(memory.after(smallvec![stored]), range.clone()));
    let result = add_implicit_barriers(loaded.end(smallvec![range]));

    assert!(matches!(result.op(), Op::End(ops::End { computation, .. })
        if matches!(computation.op(), Op::Barrier(ops::Barrier { src, deps })
            if matches!(src.op(), Op::Load(..)) && matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, src)))));
}

/// A barrier is only needed for a local buffer read and written across at least
/// two iterations: a single iteration and global memory both stay clean.
#[test_case(AddrSpace::Local, 1; "a range with a single iteration")]
#[test_case(AddrSpace::Global, 4; "global memory is not thread-shared")]
fn no_war_barrier_without_a_local_cross_iteration_hazard(addrspace: AddrSpace, extent: i64) {
    let memory = buffer(0, addrspace);
    let range = range(extent, AxisType::Weak, 0);
    let loaded = load(index_of(memory.clone(), range.clone()));
    let stored = index_of(memory, range.clone()).store(loaded);
    let result = add_implicit_barriers(stored.clone().end(smallvec![range]));

    let (computation, _) = expect_end(&result);
    assert_same!(computation, stored);
}

#[test]
fn no_war_barrier_for_a_symbolic_loop_of_possibly_zero_extent() {
    let memory = buffer(0, AddrSpace::Local);
    let range = range_symbolic(UOp::variable("n".into(), 0, 0, DType::Int32), 0);
    let loaded = load(index_of(memory.clone(), range.clone()));
    let stored = index_of(memory, range.clone()).store(loaded);
    let result = add_implicit_barriers(stored.clone().end(smallvec![range]));

    let (computation, _) = expect_end(&result);
    assert_same!(computation, stored);
}

/// A global load does not meet a local store: only the same buffer counts.
#[test]
fn unrelated_global_load_does_not_match_local_store() {
    let local = buffer(0, AddrSpace::Local);
    let global = buffer(1, AddrSpace::Global);
    let range = range(4, AxisType::Weak, 0);
    let stored = index_of(local, range.clone()).store(UOp::native_const(1.0f32));
    let loaded = load(index_of(global, range.clone()));
    let computation = UOp::sink(vec![stored, loaded]);
    let result = add_implicit_barriers(computation.clone().end(smallvec![range]));

    let (rewritten, _) = expect_end(&result);
    assert_same!(rewritten, computation);
}

/// A graph with no local STORE is returned untouched, without a rewrite walk.
#[test]
fn a_graph_without_a_local_store_is_returned_untouched() {
    let range = range(4, AxisType::Weak, 0);
    let root = load(index_of(buffer(0, AddrSpace::Global), range.clone())).end(smallvec![range]);

    assert_same!(add_implicit_barriers(root.clone()), root);
}

/// `access_buffer` follows every aliasing wrapper; a `BITCAST` is deliberately
/// not one of them.
#[test_case(0, 0, true; "a bare access on both sides")]
#[test_case(1, 0, true; "a casted store")]
#[test_case(0, 1, true; "a casted load")]
#[test_case(2, 0, true; "a reshaped store")]
#[test_case(3, 0, true; "an after-wrapped store")]
#[test_case(4, 4, true; "an mselect access on both sides")]
#[test_case(6, 6, true; "an mstack access on both sides")]
#[test_case(5, 5, false; "a bitcast is not an alias the walk follows")]
fn access_buffer_follows_aliasing_wrappers(store_kind: usize, load_kind: usize, barriered: bool) {
    let local = buffer(0, AddrSpace::Local);
    let range = range(4, AxisType::Weak, 0);
    let loaded = load(wrapped_index(load_kind, local.clone(), range.clone()));
    let stored = wrapped_index(store_kind, local, range.clone()).store(loaded);
    let result = add_implicit_barriers(stored.end(smallvec![range]));

    let (computation, _) = expect_end(&result);
    assert_eq!(matches!(computation.op(), Op::Barrier(..)), barriered, "{}", result.tree());
}

/// A load outside the loop's scope — a gather the loop's fill is ordered after
/// through `After` — is not re-run against the loop's stores, so the END gets no
/// per-iteration barrier (the double-buffer commit of a strip the K loop read).
#[test]
fn a_local_load_outside_the_loop_scope_does_not_bar_it() {
    let local = buffer(0, AddrSpace::Local);
    let gather = load(index_of(local.clone(), UOp::index_const(3)));
    let fill = range(4, AxisType::Loop, 0);
    let stored = index_of(local.after(smallvec![gather]), fill.clone()).store(UOp::native_const(1.0f32));
    let result = add_implicit_barriers(stored.end(smallvec![fill]));

    assert!(
        matches!(result.op(), Op::End(ops::End { computation, .. })
        if matches!(computation.op(), Op::Store(..))),
        "{}",
        result.tree()
    );
}

/// A store in a different buffer's scope does not bar a reduce over another one:
/// the store does not reference the inner range, so the inner END stays clean.
#[test]
fn a_local_store_outside_the_loop_scope_does_not_bar_it() {
    let local = buffer(0, AddrSpace::Local);
    let outer = range(4, AxisType::Weak, 0);
    let inner = range(4, AxisType::Weak, 1);
    let stored = index_of(local.clone(), outer).store(UOp::native_const(1.0f32));
    let loaded = load(index_of(local, inner.clone()));
    let result = add_implicit_barriers(UOp::sink(vec![stored, loaded]).end(smallvec![inner]));

    assert!(matches!(result.op(), Op::End(ops::End { computation, .. })
        if matches!(computation.op(), Op::Sink(..))));
}
