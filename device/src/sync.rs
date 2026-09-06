//! Timeline synchronization primitives for parallel execution.
//!
//! This module provides device-agnostic synchronization using timeline signals,
//! which are monotonically increasing counters that enable ordering of operations
//! across devices.
//!
//! # Design
//!
//! Timeline signals abstract over:
//! - CPU: `AtomicU64` with parking_lot condvar for waiting
//! - AMD: `AmdSignal`, a GPU-visible signal slot the command processor bumps
//!
//! # Example
//!
//! ```ignore
//! let signal = CpuTimelineSignal::new();
//!
//! // Producer thread
//! signal.set(1);  // Signal completion of operation 1
//!
//! // Consumer thread
//! signal.wait(1, 1000)?;  // Wait for operation 1 to complete
//! ```

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::error::{Result, TimelineTimeoutSnafu};
use snafu::ensure;

/// Hardware-stamped GPU dispatch timestamps, readable once the dispatch
/// retired. Backend-agnostic: AMD reads the completion signal's CP stamps,
/// Metal the command buffer's GPU times, CUDA will wrap an event pair. `None`
/// until retirement or when the backend can't stamp.
pub trait DispatchTimestamps: Send + Sync {
    /// `(start_ns, end_ns)` on the GPU clock.
    fn timestamps_ns(&self) -> Option<(u64, u64)>;

    /// Hardware performance counters harvested for this dispatch, when PMC was
    /// armed for it. `None` by default and on backends without counter support.
    fn counters(&self) -> Option<crate::profile::CounterSet> {
        None
    }
}

/// Waitable completion token for a batch of submitted device work.
///
/// Held per storage by the AMD and CUDA scoped-sync producer tables
/// (`AmdDeviceCore::wait_storage`, `CudaDevice::wait_storage`) so host reads
/// wait only the producing submissions instead of draining the whole device.
/// `wait` must be safe to call from any thread, any number of times, without
/// holding queue or lane references. `Any` lets a backend recover its own
/// token type (CUDA orders copies after the token's event on the GPU).
pub trait CompletionToken: Send + Sync + Any {
    fn wait(&self, timeout_ms: u64) -> Result<()>;
    fn retired(&self) -> bool;

    /// The owner has recorded this token on every storage it covers. Until
    /// then a backend that tracks unpublished submissions per lane keeps
    /// the token's lane flagged, so a scoped wait racing the publication
    /// still drains it. No-op by default.
    fn published(&self) {}
}

/// Monotonic timeline signal for synchronization.
///
/// Timeline signals provide a way to order operations across different execution
/// contexts (threads, devices, queues). The signal value only increases, and
/// waiters block until the signal reaches or exceeds the target value.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` for cross-thread use.
pub trait TimelineSignal: Send + Sync + std::fmt::Debug + Any {
    /// Return this signal as `Any` for checked type-erased queue dispatch.
    fn as_any(&self) -> &dyn Any;

    /// Get the current signal value.
    fn value(&self) -> u64;

    /// Set the signal to a new value.
    ///
    /// # Panics
    ///
    /// May panic if `value` is less than the current value (implementation-defined).
    fn set(&self, value: u64);

    /// Wait for the signal to reach or exceed `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - The target value to wait for
    /// * `timeout_ms` - Maximum time to wait in milliseconds (0 = infinite)
    ///
    /// # Returns
    ///
    /// `Ok(())` if the signal reached the target value, or `Err` on timeout.
    fn wait(&self, value: u64, timeout_ms: u64) -> Result<()>;

    /// Check if the signal has reached `value` without blocking.
    fn is_reached(&self, value: u64) -> bool {
        self.value() >= value
    }
}

/// CPU-based timeline signal using atomics and condvar.
///
/// Efficient for CPU-only workloads. Uses `AtomicU64` for the counter and
/// `parking_lot::Condvar` for efficient waiting.
#[derive(Debug, Clone)]
pub struct CpuTimelineSignal {
    inner: Arc<CpuTimelineSignalInner>,
}

#[derive(Debug)]
struct CpuTimelineSignalInner {
    /// Current timeline value (monotonically increasing).
    value: AtomicU64,
    /// Mutex for condvar waiting (protects nothing, just for condvar).
    mutex: Mutex<()>,
    /// Condvar for waiting threads.
    condvar: Condvar,
}

impl Default for CpuTimelineSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuTimelineSignal {
    /// Create a new CPU timeline signal starting at 0.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(CpuTimelineSignalInner {
                value: AtomicU64::new(0),
                mutex: Mutex::new(()),
                condvar: Condvar::new(),
            }),
        }
    }

    /// Create a new CPU timeline signal with an initial value.
    pub fn with_initial(initial: u64) -> Self {
        Self {
            inner: Arc::new(CpuTimelineSignalInner {
                value: AtomicU64::new(initial),
                mutex: Mutex::new(()),
                condvar: Condvar::new(),
            }),
        }
    }
}

impl TimelineSignal for CpuTimelineSignal {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn value(&self) -> u64 {
        self.inner.value.load(Ordering::Acquire)
    }

    fn set(&self, value: u64) {
        let previous = self.inner.value.fetch_max(value, Ordering::AcqRel);
        if value > previous {
            self.inner.condvar.notify_all();
        }
    }

    fn wait(&self, target: u64, timeout_ms: u64) -> Result<()> {
        // Fast path: already reached
        if self.inner.value.load(Ordering::Acquire) >= target {
            return Ok(());
        }

        let mut guard = self.inner.mutex.lock();

        if timeout_ms == 0 {
            // Infinite wait
            while self.inner.value.load(Ordering::Acquire) < target {
                self.inner.condvar.wait(&mut guard);
            }
            Ok(())
        } else {
            // Timed wait
            let deadline = Instant::now() + Duration::from_millis(timeout_ms);

            while self.inner.value.load(Ordering::Acquire) < target {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    let current = self.inner.value.load(Ordering::Acquire);
                    ensure!(
                        current >= target,
                        TimelineTimeoutSnafu { what: "timeline signal", target, current, waited_ms: timeout_ms }
                    );
                    return Ok(());
                }

                let result = self.inner.condvar.wait_for(&mut guard, remaining);
                let current = self.inner.value.load(Ordering::Acquire);
                if result.timed_out() && current < target {
                    return TimelineTimeoutSnafu { what: "timeline signal", target, current, waited_ms: timeout_ms }
                        .fail();
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
#[path = "test/unit/sync.rs"]
mod tests;
