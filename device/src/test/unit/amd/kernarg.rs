use super::test_support::{MockAmdIface, mock_device, mock_device_with_signals, scripted_error};
use crate::amd::AmdAllocator;
use crate::amd::connector::PoolQueue;
use crate::amd::iface::AllocKind;
use crate::amd::kernarg::*;
use crate::amd::va_registry::AllocTag;
use crate::error::Error;
use crate::sync::TimelineSignal;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

const SLOT: usize = 64 * 1024;
/// How long a test lets a reservation sit before it calls it parked.
const PARK: Duration = Duration::from_millis(100);

/// `count` lanes of one mock device, sharing its kernarg arena.
fn lanes(count: usize) -> (Arc<MockAmdIface>, AmdAllocator, Vec<Arc<PoolQueue>>) {
    let (iface, allocator) = mock_device_with_signals(1);
    let pools = (0..count)
        .map(|_| PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).unwrap())
        .collect();
    (iface, allocator, pools)
}

fn overlaps(a: usize, b: usize, len: usize) -> bool {
    a < b + len && b < a + len
}

#[test]
fn kernarg_ring_wraps_past_abandoned_slots() {
    let (_iface, allocator) = mock_device(1);
    let arena = KernargArena::new(&allocator, allocator.dev.core()).expect("arena");
    let half = arena.size / 2;
    let first = arena.reserve(half, 16).expect("first");
    let second = arena.reserve(half, 16).expect("second");
    assert_eq!((first.offset(), second.offset()), (0, half));
    // Dropped unpublished, the GPU never saw the first slot: the wrap reuses it
    // without waiting.
    drop(first);
    let wrapped = arena.reserve(half / 2, 16).expect("wrap");
    assert_eq!(wrapped.offset(), 0);
    assert_eq!(arena.live_slots(), [(half, "pending"), (0, "pending")]);
    assert!(matches!(arena.reserve(arena.size + 1, 16), Err(Error::AmdAllocFailed { .. })));
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

/// A failed wait on the slot a wrap reuses poisons the device and hands out
/// nothing, then or later.
#[test]
fn kernarg_failed_retire_wait_poisons_and_hands_out_nothing() {
    let (iface, allocator, pools) = lanes(1);
    let arena = pools[0].arena();
    arena.reserve(arena.size, 1).unwrap().publish(pools[0].pm4_signal(), 1);
    iface.script_wait(Err(Error::Runtime { message: "scripted kernarg retire wait".into() }));
    let allocations = iface.allocation_count();

    assert!(arena.reserve(16, 16).is_err());
    assert!(allocator.dev.is_poisoned());
    assert_eq!(arena.live_slots(), [(0, "submitted")]);
    assert!(arena.reserve(16, 16).is_err(), "a poisoned device reuses no slot");
    assert_eq!((iface.allocation_count(), iface.free_count()), (allocations, 0));
    drop(pools);
    assert_eq!(iface.free_count(), 0, "all hardware-referenced queue storage must remain quarantined");
}

/// The arena lives in VRAM on a multi-XCC part, where its zero fill flushes
/// once, and in uncached system memory elsewhere, which needs no flush.
#[test_case::test_case(1, AllocKind::UncachedGtt, 0; "one xcc")]
#[test_case::test_case(2, AllocKind::DeviceVram { executable: true }, 1; "two xccs")]
fn kernarg_arena_placement_follows_the_xcc_count(xccs: u32, kind: AllocKind, flushes: usize) {
    let (iface, allocator) = mock_device(xccs);
    iface.set_hdp_flush_register();
    let _arena = KernargArena::new(&allocator, allocator.dev.core()).expect("arena");
    assert_eq!(iface.alloc_kinds_for_tag(AllocTag::Kernarg), [kind]);
    assert_eq!(iface.call_count(|call| *call == super::test_support::MockAmdCall::HdpFlush), flushes);
}

/// Tinygrad allocates one kernarg buffer per device, not per lane.
#[test]
fn kernarg_arena_is_shared_by_every_lane_of_one_device() {
    let (iface, _allocator, mut pools) = lanes(2);
    assert_eq!(iface.alloc_count_for_tag(AllocTag::Kernarg), 1);
    assert!(std::ptr::eq(pools[0].arena(), pools[1].arena()));
    // Reservations interleave through one ring, so no two lanes share a slot.
    let (first, second) = (pools[0].arena().reserve(64, 16).unwrap(), pools[1].arena().reserve(64, 16).unwrap());
    assert!(!overlaps(first.offset(), second.offset(), 64));
    drop((first, second));

    let frees = iface.free_count();
    let second = pools.pop().unwrap();
    drop(pools);
    assert_eq!(iface.free_count(), frees + 5, "a shared arena outlives the first lane to drop");
    drop(second);
    assert_eq!(iface.alloc_count_for_tag(AllocTag::Kernarg), 1);
}

/// Lanes that find the shared arena full at once take disjoint slots once the
/// slot in their way retires. Each lane wrapping on its own handed every one of
/// them offset 0, so two lanes wrote their kernargs and AQL control programs
/// over each other: a lane that ran another's control program stored that
/// lane's timeline value instead of its own, and its next dispatch waited on
/// the CP for a value nobody writes.
#[test]
fn concurrent_wraps_hand_out_disjoint_slots() {
    const LANES: usize = 4;
    let (_iface, _allocator, pools) = lanes(1);
    let arena = pools[0].arena();
    arena.reserve(arena.size - SLOT / 2, 1).unwrap().publish(pools[0].pm4_signal(), 1);
    let slots = std::thread::scope(|scope| {
        let lanes = (0..LANES).map(|_| scope.spawn(|| arena.reserve(SLOT, 16).unwrap())).collect::<Vec<_>>();
        std::thread::sleep(PARK);
        assert!(lanes.iter().all(|lane| !lane.is_finished()), "the in-flight slot holds every wrap");
        pools[0].pm4_signal().set(1);
        lanes.into_iter().map(|lane| lane.join().unwrap()).collect::<Vec<_>>()
    });
    let mut offsets = slots.iter().map(KernargSlot::offset).collect::<Vec<_>>();
    offsets.sort_unstable();
    assert!(offsets.windows(2).all(|pair| pair[0] + SLOT <= pair[1]), "overlapping kernarg slots {offsets:?}");
}

/// Lanes reserving, writing and publishing at once while a simulated GPU reads
/// each slot back in submission order and only then retires it: across several
/// wraps of the shared ring, no slot is overwritten before its reader is done.
#[test]
fn concurrent_lanes_wrapping_never_overwrite_a_live_slot() {
    const LANES: usize = 4;
    const SIZE: usize = 4096;
    const ROUNDS: usize = 4096;
    let (_iface, _allocator, pools) = lanes(LANES);
    let arena = pools[0].arena();
    assert!(LANES * ROUNDS * SIZE > 3 * arena.size, "the ring must wrap several times");
    let base = arena.base_host.as_ptr() as usize;
    let (submit, submitted) = std::sync::mpsc::channel::<(usize, usize, u8, u64)>();
    std::thread::scope(|scope| {
        for (lane, pool) in pools.iter().enumerate() {
            let submit = submit.clone();
            scope.spawn(move || {
                for round in 0..ROUNDS {
                    let mut slot = pool.arena().reserve(SIZE, 16).unwrap();
                    let pattern = (lane * 61 + round) as u8;
                    slot.host().fill(pattern);
                    let (offset, value) = (slot.offset(), round as u64 + 1);
                    slot.publish(pool.pm4_signal(), value);
                    submit.send((lane, offset, pattern, value)).unwrap();
                }
            });
        }
        drop(submit);
        for (lane, offset, pattern, value) in submitted {
            // SAFETY: a published slot lies inside the arena; the GPU only reads it.
            let bytes = unsafe { std::slice::from_raw_parts((base + offset) as *const u8, SIZE) };
            assert!(bytes.iter().all(|&byte| byte == pattern), "lane {lane} slot {offset:#x} overwritten while live");
            pools[lane].pm4_signal().set(value);
        }
    });
}

/// A slot reserved before the ring wraps and published only after it would
/// have wrapped again stays out of every later reservation: the wrap that
/// meets it waits for its publication, then for its timeline point, and hands
/// its range to no one meanwhile.
#[test]
fn unpublished_slot_survives_wraps_until_it_retires() {
    let (_iface, _allocator, pools) = lanes(2);
    let (owner, other) = (&pools[0], &pools[1]);
    let arena = owner.arena();
    let pending = arena.reserve(SLOT, 16).unwrap();
    other.pm4_signal().set(1);
    let handed = AtomicUsize::new(0);
    let (blocked, offsets) = std::thread::scope(|scope| {
        let laps = scope.spawn(|| {
            // Twice around the ring, each slot retired as soon as it is published.
            (0..2 * arena.size / SLOT)
                .map(|_| {
                    let slot = arena.reserve(SLOT, 16).unwrap();
                    let offset = slot.offset();
                    slot.publish(other.pm4_signal(), 1);
                    handed.fetch_add(1, Ordering::SeqCst);
                    offset
                })
                .collect::<Vec<_>>()
        });
        let parked = |what: &str| {
            std::thread::sleep(PARK);
            let before = handed.load(Ordering::SeqCst);
            std::thread::sleep(PARK);
            assert!(!laps.is_finished(), "the ring passed a {what} slot");
            assert_eq!(handed.load(Ordering::SeqCst), before, "the ring passed a {what} slot");
        };
        parked("pending");
        assert_eq!(arena.live_slots()[0], (0, "pending"));
        let blocked = handed.load(Ordering::SeqCst);
        pending.publish(owner.pm4_signal(), 1);
        parked("submitted");
        owner.pm4_signal().set(1);
        (blocked, laps.join().unwrap())
    });
    assert_eq!(blocked, arena.size / SLOT - 1, "the first lap fills every range but the live one");
    assert!(offsets[..blocked].iter().all(|&offset| !overlaps(offset, 0, SLOT)), "{offsets:?}");
    assert_eq!(offsets[blocked], 0, "the retired range is the next one reused");
}

/// The wrap waits for the slot it reuses and nothing more: another lane whose
/// work never retires, holding a later slot, does not delay it.
#[test]
fn wrap_waits_only_for_the_slots_it_reuses() {
    let (_iface, _allocator, pools) = lanes(2);
    let (needed, unrelated) = (&pools[0], &pools[1]);
    let arena = needed.arena();
    arena.reserve(SLOT, 16).unwrap().publish(needed.pm4_signal(), 1);
    arena.reserve(SLOT, 16).unwrap().publish(unrelated.pm4_signal(), 1);
    drop(arena.reserve(arena.size - 2 * SLOT, 16).unwrap());
    let wrapped = std::thread::scope(|scope| {
        let wrap = scope.spawn(|| arena.reserve(SLOT / 2, 16).map(|slot| slot.offset()));
        std::thread::sleep(PARK);
        assert!(!wrap.is_finished(), "the wrap must wait for the slot it reuses");
        needed.pm4_signal().set(1);
        wrap.join().unwrap()
    });
    assert_eq!(wrapped.unwrap(), 0);
    assert_eq!(unrelated.pm4_signal().value(), 0, "the unrelated lane never retired");
    assert_eq!(arena.live_slots()[0], (SLOT, "submitted"));
}

/// A reservation parked on another lane's pending slot reports the device's
/// poison when that lane's submission faults, instead of waiting it out.
#[test]
fn reservation_parked_on_a_pending_slot_sees_the_poison() {
    let (_iface, allocator, pools) = lanes(1);
    let arena = pools[0].arena();
    let pending = arena.reserve(arena.size, 1).unwrap();
    let parked = std::thread::scope(|scope| {
        let parked = scope.spawn(|| arena.reserve(16, 16).map(|slot| slot.offset()));
        std::thread::sleep(PARK);
        assert!(!parked.is_finished());
        pending.publish(pools[0].pm4_signal(), 1);
        allocator.dev.core().poison("scripted submission fault");
        parked.join().unwrap()
    });
    assert!(matches!(parked, Err(Error::Runtime { message }) if message == "scripted submission fault"));
    assert_eq!(arena.live_slots(), [(0, "submitted")]);
}

/// A slot whose lane never publishes it bounds the wait with a typed timeout.
#[test]
fn reservation_times_out_on_a_slot_never_published() {
    let (_iface, allocator, pools) = lanes(1);
    let arena = pools[0].arena();
    let _pending = arena.reserve(arena.size, 1).unwrap();
    let error = arena.reserve(16, 16).map(|slot| slot.offset()).unwrap_err();
    assert!(matches!(error, Error::TimelineTimeout { what: "AMD kernarg slot publication", .. }), "{error:?}");
    assert!(!allocator.dev.is_poisoned(), "a slow publisher says nothing about the GPU");
}

/// A lane's timeline reset lowers its signal below the values its slots
/// reached; the drain before it retires those slots for good.
#[test]
fn drained_timeline_retires_its_slots_across_a_reset() {
    let (_iface, _allocator, pools) = lanes(2);
    let arena = pools[0].arena();
    arena.reserve(SLOT, 16).unwrap().publish(pools[0].pm4_signal(), 7);
    arena.reserve(SLOT, 16).unwrap().publish(pools[1].pm4_signal(), 7);
    arena.retire_timeline(pools[0].pm4_signal());
    assert_eq!(arena.live_slots(), [(0, "done"), (SLOT, "submitted")]);
}
