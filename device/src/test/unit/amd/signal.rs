use super::test_support::{MockAmdCall, MockAmdIface, amd_alloc_or_skip, mock_device};
use crate::amd::AmdAllocator;
use crate::amd::connector::SubmissionFinalizer;
use crate::amd::signal::*;
use crate::error::Error;
use crate::sync::TimelineSignal;
use std::sync::Arc;

fn mock_signal() -> (Arc<MockAmdIface>, AmdAllocator, Arc<SignalPool>, Arc<AmdSignal>) {
    let (iface, allocator) = mock_device(1);
    let pool = SignalPool::new(&allocator, 1).expect("mock signal pool");
    let signal = Arc::new(pool.acquire().expect("mock signal"));
    (iface, allocator, pool, signal)
}

fn wait_count(iface: &MockAmdIface) -> usize {
    iface.call_count(|call| matches!(call, MockAmdCall::WaitEvents { .. }))
}

/// Live pool round-trip: concurrent slots are distinct, and a released slot is
/// handed back out with its value zeroed.
#[test]
fn signal_pool_slots_are_distinct_and_reset_before_reuse() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let pool = SignalPool::new(&alloc, 64).expect("create pool");
    let first = pool.acquire().expect("acquire");
    let second = pool.acquire().expect("acquire");
    assert_ne!(first.value_addr(), second.value_addr());
    assert_eq!(first.value(), 0);
    first.set(7);
    assert_eq!(first.value(), 7);

    let slot = first.slot();
    drop(first);
    let reused = pool.acquire().expect("reacquire");
    assert_eq!(reused.slot(), slot);
    assert_eq!(reused.value(), 0, "a reused slot must not carry the old value");
}

#[test]
fn signal_slot_releases_once_after_last_finalizer_clone_drops() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let pool = SignalPool::new(&alloc, 64).expect("create pool");
    let free = pool.free();
    let signal = Arc::new(pool.acquire().expect("acquire"));
    signal.reset(0);
    let finalizer = SubmissionFinalizer::timeline(signal, 1, None);
    let clone = Arc::clone(&finalizer);
    assert_eq!(pool.free(), free - 1);
    drop(finalizer);
    assert_eq!(pool.free(), free - 1, "a retained finalizer must keep its completion slot");
    drop(clone);
    assert_eq!(pool.free(), free, "the finalizer's last drop releases exactly once");
}

/// A backend wait failure surfaces verbatim and latches the device; afterwards
/// every wait fails on the latch without reaching the backend again.
#[test]
fn wait_events_error_stays_typed_poisons_owner_and_short_circuits_later_waits() {
    let (iface, allocator, _pool, signal) = mock_signal();
    iface.script_wait(Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_WAIT_EVENTS", errno: 5 }));

    let error = signal.wait(1, 10_000).expect_err("scripted wait failure");
    assert!(matches!(error, Error::AmdIoctl { ioctl: "AMDKFD_IOC_WAIT_EVENTS", errno: 5 }));
    assert!(allocator.dev.is_poisoned());
    assert!(
        matches!(allocator.dev.poison_error(), Some(Error::Runtime { message }) if message.contains("WAIT_EVENTS"))
    );
    let waits = wait_count(&iface);
    assert_eq!(waits, 1);

    let error = signal.wait(1, 10_000).expect_err("poison must fail before polling");
    assert!(matches!(error, Error::Runtime { message } if message.contains("WAIT_EVENTS")));
    assert_eq!(wait_count(&iface), waits, "a later wait reached the backend despite the poison latch");
}

#[test]
fn poisoned_owner_wakes_active_signal_waiter() {
    let (iface, allocator, _pool, signal) = mock_signal();
    let (tx, rx) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || tx.send(signal.wait(1, 60_000)).unwrap());

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while wait_count(&iface) == 0 {
        assert!(std::time::Instant::now() < deadline, "waiter never entered event polling");
        std::thread::yield_now();
    }
    allocator.dev.poison("concurrent synthetic fault");
    let error = rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("poisoned waiter did not wake")
        .expect_err("poisoned waiter must fail");
    assert!(matches!(error, Error::Runtime { message } if message == "concurrent synthetic fault"));
    waiter.join().unwrap();
}

#[test]
fn signal_pool_construction_failure_and_drop_balance_backing() {
    let (iface, allocator) = mock_device(1);
    iface.script_alloc(Err(Error::Runtime { message: "scripted signal allocation".into() }));
    assert!(SignalPool::new(&allocator, 64).is_err());
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (0, 0, 0));

    let pool = SignalPool::new(&allocator, 64).unwrap();
    assert_eq!(iface.live_handle_count(), 1);
    drop(pool);
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (1, 1, 0));
    assert!(iface.free_issues().is_empty());
}

#[test]
fn signal_pool_grows_a_chunk_and_releases_unwound_slots() {
    let (iface, allocator) = mock_device(1);
    let pool = SignalPool::new(&allocator, 64).unwrap();
    let capacity = pool.capacity();
    let held = (0..capacity).map(|_| pool.acquire().expect("slot")).collect::<Vec<_>>();
    assert_eq!((pool.free(), iface.allocation_count()), (0, 1));

    // Exhaustion carves another chunk instead of erroring (tinygrad
    // `HCQCompiled.new_signal`, support/hcq.py:452-458).
    let extra = pool.acquire().expect("exhausted pool must grow");
    assert_eq!((pool.capacity(), iface.allocation_count(), pool.free()), (capacity * 2, 2, capacity - 1));

    // A panic unwind still returns its slot; only a poisoned device retains one.
    let free_before = pool.free();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _signal = pool.acquire().expect("slot");
        panic!("scripted signal abandonment");
    }));
    assert!(result.is_err());
    assert_eq!(pool.free(), free_before);

    drop((extra, held));
    assert_eq!(pool.free(), pool.capacity());
}

#[test]
fn prepared_finalizer_wait_is_bounded_by_its_deadline() {
    let (_iface, _allocator, _pool, signal) = mock_signal();
    let finalizer = SubmissionFinalizer::prepared_timeline(signal, 1, Vec::new());
    let started = std::time::Instant::now();
    let error = finalizer.wait(50).expect_err("an unpublished submission must not park forever");
    assert!(matches!(error, Error::TimelineTimeout { what: "AMD submission publication", .. }), "{error:?}");
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
}

/// A wait gives up on a queue that stops moving, and not on one whose read
/// index keeps moving past the limit.
#[test]
fn a_moving_read_pointer_keeps_a_wait_alive() {
    let (_iface, _allocator, _pool, signal) = mock_signal();
    let index = Box::leak(Box::new(std::sync::atomic::AtomicU64::new(0)));
    let read = QueueReadPointer::new(std::ptr::NonNull::new(index.as_ptr()).expect("a leaked box"));
    signal.watch_queue(read);
    assert!(signal.wait_signal_value(1, 20).is_err(), "the index stands still");
    let reader = &*index;
    let done = Arc::clone(&signal);
    let mover = std::thread::spawn(move || {
        for _ in 0..30 {
            reader.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        done.reset(1);
    });
    signal.wait_signal_value(1, 50).expect("the index keeps it alive");
    mover.join().unwrap();
}
