use super::test_support::{amd_alloc_or_skip, mock_device, mock_device_with_signals, scripted_error};
use crate::amd::connector::PoolQueue;
use crate::amd::kernarg::*;
use crate::amd::va_registry::AllocTag;
use crate::error::Error;
use std::sync::Arc;

#[test]
fn kernarg_bump_wraps_when_full() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let arena = KernargArena::new(&alloc, alloc.dev.core()).expect("arena");
    let half = arena.size / 2;
    let first = arena.bump(half, 16).expect("first");
    assert_eq!(first, 0);
    let second = arena.bump(half / 2, 16).expect("second");
    assert!(second > first && second < arena.size);
    // A request that would overflow drains every live connector (a no-op on an
    // idle device) and resets the cursor to the start of the arena.
    assert_eq!(arena.bump(arena.size - 16, 16).expect("wrap"), 0);
}

#[test]
fn kernarg_arena_construction_balances_and_scripts_allocation_failure() {
    let (iface, allocator) = mock_device(1);
    iface.script_alloc(Err(scripted_error("kernarg allocation")));
    assert!(KernargArena::new(&allocator, allocator.dev.core()).is_err());
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (0, 0, 0));

    let arena = KernargArena::new(&allocator, allocator.dev.core()).expect("arena");
    assert_eq!(iface.allocation_count(), 1);
    drop(arena);
    assert_eq!((iface.free_count(), iface.live_handle_count()), (1, 0));
    assert!(iface.free_issues().is_empty());
}

#[test]
fn kernarg_wrap_failed_drain_does_not_reset_or_free() {
    let (iface, allocator) = mock_device_with_signals(1);
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).unwrap();
    pool.arena().bump(pool.arena().size, 1).unwrap();
    pool.next_pm4();
    iface.script_wait(Err(Error::Runtime { message: "scripted kernarg wrap drain".into() }));
    let allocations = iface.allocation_count();

    assert!(pool.arena().bump(16, 16).is_err());
    assert!(allocator.dev.is_poisoned());
    assert_eq!((iface.allocation_count(), iface.free_count()), (allocations, 0));
    drop(pool);
    assert_eq!(iface.free_count(), 0, "all hardware-referenced queue storage must remain quarantined");
}

/// Tinygrad allocates one kernarg buffer per device, not per lane.
#[test]
fn kernarg_arena_is_shared_by_every_lane_of_one_device() {
    let (iface, allocator) = mock_device_with_signals(1);
    let first = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).unwrap();
    let second = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).unwrap();
    assert_eq!(iface.alloc_count_for_tag(AllocTag::Kernarg), 1);
    assert!(std::ptr::eq(first.arena(), second.arena()));
    // Bumps interleave through one cursor, so no two lanes share a slot.
    assert_ne!(first.arena().bump(64, 16).unwrap(), second.arena().bump(64, 16).unwrap());

    let frees = iface.free_count();
    drop(first);
    assert_eq!(iface.free_count(), frees + 5, "a shared arena outlives the first lane to drop");
    drop(second);
    assert_eq!(iface.alloc_count_for_tag(AllocTag::Kernarg), 1);
}

/// Lanes that find the shared arena full at once wrap it once between them.
/// Each wrapping on its own handed every one of them offset 0, so two lanes
/// wrote their kernargs and AQL control programs over each other: a lane that
/// ran another's control program stored that lane's timeline value instead of
/// its own, and its next dispatch waited on the CP for a value nobody writes.
#[test]
fn concurrent_wraps_hand_out_disjoint_slots() {
    use crate::sync::TimelineSignal;
    const LANES: usize = 4;
    const SLOT: usize = 256;
    let (_iface, allocator) = mock_device_with_signals(1);
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).unwrap();
    let arena = pool.arena();
    arena.bump(arena.size - SLOT / 2, 1).unwrap();
    // An in-flight dispatch holds the wrap's drain open until it retires, so
    // every lane meets the full arena while the first wrap is still draining.
    let in_flight = pool.next_pm4();
    let mut slots = std::thread::scope(|scope| {
        let lanes = (0..LANES).map(|_| scope.spawn(|| arena.bump(SLOT, 16).unwrap())).collect::<Vec<_>>();
        std::thread::sleep(std::time::Duration::from_millis(100));
        pool.pm4_signal().set(in_flight);
        lanes.into_iter().map(|lane| lane.join().unwrap()).collect::<Vec<_>>()
    });
    slots.sort_unstable();
    assert!(slots.windows(2).all(|pair| pair[0] + SLOT <= pair[1]), "overlapping kernarg slots {slots:?}");
}
