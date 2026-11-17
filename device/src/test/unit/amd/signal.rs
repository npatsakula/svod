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

/// A completion signal counts the CP's decrements from zero, and with a
/// mailbox the decrement raises the queue event: the CP writes `event_id` to
/// `event_mailbox_ptr` (`amd_signal_t` +16 / +24) and interrupts.
#[test_case::test_case(None; "silent")]
#[test_case::test_case(Some(crate::amd::iface::QueueEventMailbox { address: 0xfeed_0000, event_id: 9 }); "interrupting")]
fn a_completion_signal_counts_decrements(mailbox: Option<crate::amd::iface::QueueEventMailbox>) {
    let (_iface, allocator) = mock_device(1);
    let pool = SignalPool::new(&allocator, 1).expect("mock signal pool");
    let signal = pool.acquire().expect("mock signal").counting_down(mailbox);
    // SAFETY: the mock GPU VA is the host address of the 64-byte slot.
    let word = |offset: u64| unsafe { std::ptr::read_volatile((signal.handle() + offset) as *const u64) };
    assert_eq!(signal.value_addr(), signal.handle() + 8);
    assert_eq!(word(16), mailbox.map_or(0, |mailbox| mailbox.address));
    assert_eq!(word(24) as u32, mailbox.map_or(0, |mailbox| mailbox.event_id));
    assert_eq!(signal.value(), 0);
    signal.complete_packet();
    signal.complete_packet();
    assert_eq!((signal.value(), word(8)), (2, (-2i64) as u64));
    signal.wait_signal_value(2, 1).expect("two retired packets");
    signal.reset(0);
    assert_eq!(word(8), 0);
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
    assert_eq!(iface.counts(), (0, 0, 0));

    let pool = SignalPool::new(&allocator, 64).unwrap();
    assert_eq!(iface.counts(), (1, 0, 1));
    drop(pool);
    assert_eq!(iface.counts(), (1, 1, 0));
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

/// A wait sleeps in `WAIT_EVENTS` once it outlives its polling budget, even
/// while the queue keeps moving, and a silent one times out no sooner than its
/// limit.
#[test]
fn a_long_wait_sleeps_while_the_queue_moves() {
    let (iface, _allocator, _pool, signal) = mock_signal();
    let index = Box::leak(Box::new(std::sync::atomic::AtomicU64::new(0)));
    signal.watch_queue(QueueReadPointer::new(std::ptr::NonNull::new(index.as_ptr()).expect("a leaked box")));

    let started = std::time::Instant::now();
    let error = signal.wait_signal_value(1, 20).expect_err("the index stands still");
    assert!(matches!(error, Error::TimelineTimeout { .. }), "{error:?}");
    assert!(started.elapsed() >= std::time::Duration::from_millis(20));

    let reader = &*index;
    let done = Arc::clone(&signal);
    let mover = std::thread::spawn(move || {
        // Faster than the polling budget, so only the wait's age can sleep it.
        for _ in 0..300 {
            reader.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        done.reset(1);
    });
    let sleeps = wait_count(&iface);
    signal.wait_signal_value(1, 20).expect("the index keeps it alive");
    mover.join().unwrap();
    assert!(wait_count(&iface) > sleeps, "a moving queue kept the wait polling");
}

/// Which error word a test plays the CP or KFD writing.
#[derive(Clone, Copy, Debug)]
enum QueueReport {
    Inactive,
    ErrorReason,
}

/// Write `code` into `lane`'s `report` word, as the CP or KFD does.
fn report_queue_error(lane: &crate::amd::connector::PoolQueue, report: QueueReport, code: u64) {
    let (inactive, reason) = lane.queue().error_word_ptrs();
    let word = match report {
        QueueReport::Inactive => inactive,
        QueueReport::ErrorReason => reason,
    };
    // SAFETY: the word lives in the lane's error allocation, alive with `lane`.
    unsafe { std::ptr::write_volatile(word, code) };
}

/// A wait sleeping in `WAIT_EVENTS` wakes on the queue event a stopped queue
/// raises and fails at once with the decoded error, which poisons the device
/// for every later wait.
#[test_case::test_case(2, QueueReport::Inactive, 0x401, "queue_inactive_signal", "insufficient scratch"; "aql halt")]
#[test_case::test_case(2, QueueReport::ErrorReason, 1 << 29, "ErrorReason", "preemption error"; "aql exception")]
#[test_case::test_case(1, QueueReport::ErrorReason, 1 << 3, "ErrorReason", "illegal instruction"; "pm4 exception")]
fn a_stopped_queue_fails_a_blocked_wait_at_once(
    xccs: u32,
    report: QueueReport,
    code: u64,
    expected_report: &str,
    expected_cause: &str,
) {
    let (iface, allocator) = super::test_support::mock_device_with_signals(xccs);
    let lane = super::test_support::pool_queue(&allocator);
    let signal = Arc::clone(lane.pm4_signal());
    let waiting = Arc::clone(&signal);
    let (tx, rx) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || tx.send(waiting.wait(1, 60_000)).unwrap());

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while wait_count(&iface) == 0 {
        assert!(std::time::Instant::now() < deadline, "waiter never entered event polling");
        std::thread::yield_now();
    }
    report_queue_error(&lane, report, code);
    let error = rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("the stopped queue did not wake the waiter")
        .expect_err("a stopped queue must fail the wait");
    waiter.join().unwrap();
    let Error::AmdQueueError { report, code: reported, cause, .. } = error else { panic!("{error:?}") };
    assert_eq!((report, reported, cause), (expected_report, code, expected_cause));

    assert!(allocator.dev.is_poisoned());
    let later = signal.wait(1, 60_000).expect_err("the poison fails later waits");
    assert!(matches!(later, Error::Runtime { message } if message.contains(expected_cause)));
}

/// A wait that runs out its deadline names the stopped queue rather than the
/// deadline.
#[test]
fn a_timed_out_wait_names_the_stopped_queue() {
    let (_iface, allocator) = super::test_support::mock_device_with_signals(2);
    let lane = super::test_support::pool_queue(&allocator);
    report_queue_error(&lane, QueueReport::Inactive, 0x80);
    let error = lane.pm4_signal().wait(1, 1).expect_err("a stopped queue must fail the wait");
    assert!(matches!(error, Error::AmdQueueError { cause: "out of VGPRs", .. }), "{error:?}");
}

/// Zeroed error words are no report: the wait sleeps and wakes in
/// `WAIT_EVENTS` until its deadline and leaves the device usable.
#[test_case::test_case(1; "pm4")]
#[test_case::test_case(2; "aql")]
fn quiet_error_words_leave_a_wait_to_its_deadline(xccs: u32) {
    let (iface, allocator) = super::test_support::mock_device_with_signals(xccs);
    let lane = super::test_support::pool_queue(&allocator);
    let error = lane.pm4_signal().wait(1, 20).expect_err("nothing completes the wait");
    assert!(matches!(error, Error::TimelineTimeout { .. }), "{error:?}");
    assert!(wait_count(&iface) > 0, "the wait never slept, so never read the words on a wake");
    assert!(!allocator.dev.is_poisoned());
}
