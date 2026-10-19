//! Event-based synchronization: the per-plan stream context, dispatch
//! timestamps from event pairs, and completion tokens.

use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use super::cupti;
use super::device::{CudaDevice, CudaEvent, CudaStream, Lane};
use super::program::CudaProgram;
use crate::device::{PlanContext, Program};
use crate::profile::{CounterSet, CudaCounter, PmcCounter};
use crate::sync::{CompletionToken, DispatchTimestamps};
use crate::{Error, Result};

/// One plan's lane: a non-blocking stream that dispatches in submission order.
pub struct CudaPlanCtx {
    stream: CudaStream,
    /// PMC: hardware counters to collect on profiling dispatches (empty = off).
    pmc: Mutex<Vec<CudaCounter>>,
    /// The range-profiling session, opened on the first counted dispatch and
    /// reused (enabling is per-context, not per-range).
    session: Mutex<Option<cupti::Session>>,
}

impl CudaPlanCtx {
    pub fn new(dev: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self { stream: CudaStream::new(dev)?, pmc: Mutex::new(Vec::new()), session: Mutex::new(None) })
    }

    /// Capture counters around one launch.
    ///
    /// CUPTI's range profiler is host-driven: the counter data is only readable
    /// after the launch retires and the session must be stopped before the next
    /// one starts, so a counted dispatch synchronizes here rather than deferring
    /// to the plan's own sync. Range profiling serializes the context anyway.
    /// A CUPTI failure degrades this dispatch to timing only.
    fn with_counters(
        &self,
        counters: &[CudaCounter],
        launch: impl FnOnce() -> Result<()>,
    ) -> Result<Option<CounterSet>> {
        let dev = self.stream.device();
        let mut guard = self.session.lock();
        let session = match guard.as_mut() {
            Some(session) => session,
            None => match cupti::Session::new(dev.context_raw(), dev.device_id(), counters) {
                Ok(session) => guard.insert(session),
                Err(error) => {
                    tracing::warn!(%error, "CUPTI session unavailable; reporting timing only");
                    launch()?;
                    return Ok(None);
                }
            },
        };
        if let Err(error) = session.start() {
            tracing::warn!(%error, "CUPTI could not arm counters; reporting timing only");
            launch()?;
            return Ok(None);
        }
        // The launch and the sync must both happen before `stop`, even on
        // failure, or the session stays armed and leaks into the next dispatch.
        let launched = launch().and_then(|()| self.stream.synchronize());
        let captured = session.stop();
        launched?;
        match captured {
            Ok(set) => Ok(Some(set)),
            Err(error) => {
                tracing::warn!(%error, "CUPTI counter readback failed; reporting timing only");
                Ok(None)
            }
        }
    }
}

impl PlanContext for CudaPlanCtx {
    unsafe fn dispatch(
        &self,
        program: &dyn Program,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
        profile: bool,
    ) -> Result<Option<Arc<dyn DispatchTimestamps>>> {
        let program = program.as_any().downcast_ref::<CudaProgram>().ok_or_else(|| Error::ProgramAbiMismatch {
            reason: format!("CudaPlanCtx dispatched non-CUDA program {:?}", program.name()),
        })?;
        let dev = self.stream.device();
        let lane = self.stream.lane();
        dev.order_launch(lane)?;
        lane.mark_unpublished();
        let start = profile.then(|| self.stream.record(true)).transpose()?;
        // SAFETY: forwarded contract.
        let launch = || unsafe { program.launch(lane.raw(), buffers, vals, global_size, local_size) };
        let counters = match self.pmc.lock().clone() {
            armed if profile && !armed.is_empty() => self.with_counters(&armed, launch)?,
            _ => {
                launch()?;
                None
            }
        };
        let Some(start) = start else { return Ok(None) };
        let end = self.stream.record(true)?;
        Ok(Some(Arc::new(CudaDispatchTimestamps { dev: Arc::clone(dev), start, end, counters })))
    }

    fn order_after(&self, accesses: &[(&crate::Buffer, bool)]) -> Result<()> {
        self.stream.order_after(accesses)
    }

    fn completion_token(&self) -> Option<Arc<dyn CompletionToken>> {
        Some(Arc::new(self.stream.token().ok()?))
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    fn set_pmc(&self, counters: &[PmcCounter]) -> usize {
        let mut armed = self.pmc.lock();
        *armed = counters
            .iter()
            .filter_map(|c| match c {
                PmcCounter::Cuda(c) => Some(*c),
                _ => None,
            })
            .collect();
        // The session pins one metric set at construction, so a changed
        // selection needs a fresh one.
        *self.session.lock() = None;
        armed.len()
    }

    fn pmc_available(&self) -> bool {
        cupti::available(self.stream.device().context_raw())
    }

    fn pmc_default(&self) -> Vec<PmcCounter> {
        CudaCounter::all().into_iter().map(PmcCounter::Cuda).collect()
    }
}

/// GPU-clock stamps of one dispatch: an event pair around the launch. The
/// duration comes from `elapsed(start, end)` at full event resolution; the
/// absolute position is `elapsed(base, start)`, whose `f32` milliseconds
/// coarsen as the process ages, so `end` is derived from `start` to keep
/// the pair ordered and the duration exact.
pub struct CudaDispatchTimestamps {
    dev: Arc<CudaDevice>,
    start: Arc<CudaEvent>,
    end: Arc<CudaEvent>,
    /// Counters captured around this dispatch, decoded at capture time because
    /// CUPTI's readback is host-driven and cannot outlive its session.
    counters: Option<CounterSet>,
}

impl CudaDispatchTimestamps {
    pub(crate) fn new(dev: Arc<CudaDevice>, start: Arc<CudaEvent>, end: Arc<CudaEvent>) -> Self {
        Self { dev, start, end, counters: None }
    }
}

impl std::fmt::Debug for CudaDispatchTimestamps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDispatchTimestamps").field("stamps", &self.timestamps_ns()).finish()
    }
}

impl DispatchTimestamps for CudaDispatchTimestamps {
    fn timestamps_ns(&self) -> Option<(u64, u64)> {
        if !(self.start.completed().ok()? && self.end.completed().ok()?) {
            return None;
        }
        let ns = |ms: f32| (f64::from(ms.max(0.0)) * 1e6).round() as u64;
        let start = ns(self.start.elapsed_ms_since(self.dev.base_event()).ok()?);
        let duration = ns(self.end.elapsed_ms_since(self.start.raw()).ok()?);
        Some((start, start + duration))
    }

    fn counters(&self) -> Option<CounterSet> {
        self.counters.clone()
    }
}

/// One event recorded on one lane: retired once the lane reached the record.
/// The lane id lets the device's producer table keep only the newest token
/// per lane and order copies after the event on the GPU. A token minted by
/// a [`CudaStream`] also covers the lane's submissions up to that point and
/// publishes them once recorded on their storages.
#[derive(Clone)]
pub struct CudaCompletionToken {
    event: Arc<CudaEvent>,
    lane: u64,
    covers: Option<(Weak<Lane>, u64)>,
}

impl CudaCompletionToken {
    pub fn new(event: Arc<CudaEvent>, lane: u64) -> Self {
        Self { event, lane, covers: None }
    }

    pub(crate) fn covering(mut self, lane: &Arc<Lane>, seq: u64) -> Self {
        self.covers = Some((Arc::downgrade(lane), seq));
        self
    }

    pub(crate) fn event(&self) -> &Arc<CudaEvent> {
        &self.event
    }

    pub(crate) fn lane(&self) -> u64 {
        self.lane
    }
}

impl CompletionToken for CudaCompletionToken {
    fn wait(&self, timeout_ms: u64) -> Result<()> {
        self.event.wait(timeout_ms)
    }

    fn retired(&self) -> bool {
        self.event.completed().unwrap_or(true)
    }

    fn published(&self) {
        if let Some((lane, seq)) = &self.covers
            && let Some(lane) = lane.upgrade()
        {
            lane.publish(*seq);
        }
    }
}
