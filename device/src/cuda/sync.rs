//! Event-based synchronization: the per-plan stream context, dispatch
//! timestamps from event pairs, and completion tokens.

use std::sync::{Arc, Weak};

use super::device::{CudaDevice, CudaEvent, CudaStream, Lane};
use super::program::CudaProgram;
use crate::device::{PlanContext, Program};
use crate::sync::{CompletionToken, DispatchTimestamps};
use crate::{Error, Result};

/// One plan's lane: a non-blocking stream that dispatches in submission order.
pub struct CudaPlanCtx {
    stream: CudaStream,
}

impl CudaPlanCtx {
    pub fn new(dev: Arc<CudaDevice>) -> Result<Self> {
        Ok(Self { stream: CudaStream::new(dev)? })
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
        unsafe { program.launch(lane.raw(), buffers, vals, global_size, local_size) }?;
        let Some(start) = start else { return Ok(None) };
        let end = self.stream.record(true)?;
        Ok(Some(Arc::new(CudaDispatchTimestamps { dev: Arc::clone(dev), start, end })))
    }

    fn completion_token(&self) -> Option<Arc<dyn CompletionToken>> {
        Some(Arc::new(self.stream.token().ok()?))
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
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
}

impl CudaDispatchTimestamps {
    pub(crate) fn new(dev: Arc<CudaDevice>, start: Arc<CudaEvent>, end: Arc<CudaEvent>) -> Self {
        Self { dev, start, end }
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
