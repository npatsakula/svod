//! The opened CUDA device: primary context, attribute limits, the copy and
//! dispatch lanes, the base event that zeroes the GPU-clock timeline, the
//! poison latch, and the scoped-synchronization tables that let host access
//! wait only on the submissions that touched a storage.
//!
//! # Ordering model
//!
//! Every submission runs on an in-order stream (a [`Lane`]): one per plan
//! context, one per graph, the device's dispatch lane for per-call
//! `Program::execute`, and the copy lane for allocator copies and memsets.
//! Lanes are not ordered against each other by the driver, so the device
//! keeps three tables:
//!
//! - `producers`: storage base → the newest completion token per lane that
//!   read or wrote it (a host overwrite is a WAR hazard against in-flight
//!   readers too). Published by the executor after every plan execute and by
//!   the allocator after every copy-lane operation. A storage absent from the
//!   table has unknown producers and falls back to a context drain.
//! - `lanes`: every live lane and how many submissions it holds that no
//!   token has been published for yet; such lanes are waited by every
//!   scoped wait. A token counts as published only once its owner recorded
//!   it on every storage ([`CompletionToken::published`]), so the window
//!   between minting and recording is covered.
//! - `copy_tail`: the newest copy-lane event; every launch on any lane waits
//!   it on the GPU, so an asynchronous copy or memset is ordered before all
//!   later kernels without a host wait.
//!
//! `SVOD_CUDA_SCOPED_SYNC=0` disables all of it: every wait drains the
//! context and every copy synchronizes the copy stream, as before.

use std::any::Any;
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int};
use std::ptr::{NonNull, null_mut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, OnceLock, Weak};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use svod_dtype::CudaArch;

use super::sync::CudaCompletionToken;
use super::sys::{
    Api, CU_EVENT_DEFAULT, CU_EVENT_DISABLE_TIMING, CU_MEMHOSTALLOC_PORTABLE, CU_STREAM_NON_BLOCKING, CUcontext,
    CUdevice, CUdeviceptr, CUevent, CUresult, CUstream, api, attribute,
};
use crate::error::{Error, Result, TimelineTimeoutSnafu};
use crate::sync::CompletionToken;

static DEVICE_CACHE: LazyLock<Mutex<HashMap<usize, Arc<CudaDevice>>>> = LazyLock::new(Default::default);
static HAS_DEVICES: OnceLock<bool> = OnceLock::new();
/// Process-unique lane ids: a destroyed stream's handle may be reused by the
/// driver, its lane id never is.
static NEXT_LANE: AtomicU64 = AtomicU64::new(1);

/// Bounce buffer for large pageable host ↔ device copies (`cuMemcpy*Async`
/// needs pinned host memory to be asynchronous); transfers up to this size
/// go through one synchronous `cuMemcpy` instead.
pub(crate) const STAGING_BYTES: usize = 4 << 20;
/// Poll cadence of timed event waits; the driver offers no timed wait.
const EVENT_POLL: Duration = Duration::from_micros(200);
/// Producer lists longer than this are pruned of retired tokens on insert;
/// below it, the per-lane replacement keeps them short without any query.
const PRUNE_PRODUCERS_ABOVE: usize = 8;

/// In-flight tokens of one storage: at most one per lane, since a lane is
/// in order and its newest token implies the older ones.
type Producers = smallvec::SmallVec<[CudaCompletionToken; 2]>;

/// Whether the driver loads, initializes, and reports at least one device.
/// Memoized; never panics; `false` on any failure.
pub fn has_devices() -> bool {
    *HAS_DEVICES.get_or_init(|| api().and_then(|api| api.init().and_then(|()| api.device_count())).is_ok_and(|n| n > 0))
}

/// Static device limits (`cuDeviceGetAttribute`): the launch-bound check,
/// the optimizer profile's shared-memory budget and
/// [`crate::KernelResources`] occupancy.
#[derive(Debug, Clone, Copy)]
pub struct CudaLimits {
    pub sm_count: u32,
    pub max_threads_per_block: u32,
    pub max_threads_per_sm: u32,
    pub shared_per_block: u32,
    pub warp_size: u32,
    /// `cuMemAllocManaged` is usable and host access is coherent with running
    /// kernels: the backing of host-visible buffers.
    pub managed_memory: bool,
}

/// One in-order stream plus the scoped-sync state of its submissions. Owned
/// by the device (dispatch and copy lanes) or a [`CudaStream`]; the device's
/// lane registry holds it weakly, so an upgraded handle keeps the stream
/// alive across a wait even while its owner is being dropped.
pub struct Lane {
    api: &'static Api,
    /// The owning device's context, made current before the stream is destroyed.
    context: CUcontext,
    raw: CUstream,
    id: u64,
    /// Submissions so far; a token minted after the `n`th covers `n`.
    submitted: AtomicU64,
    /// Submissions covered by a published token or a drain. Every scoped
    /// wait drains a lane whose count trails `submitted`.
    published: AtomicU64,
    /// `CudaDevice::copy_seq` value this lane last waited on.
    copies_seen: AtomicU64,
}

// SAFETY: the driver's stream handle is thread-safe; the flags are atomic.
unsafe impl Send for Lane {}
unsafe impl Sync for Lane {}

impl Lane {
    fn create(api: &'static Api, context: CUcontext) -> Result<Arc<Self>> {
        let mut raw = CUstream::NULL;
        // SAFETY: out-pointer to a live handle slot.
        unsafe { (api.stream_create)(&mut raw, CU_STREAM_NON_BLOCKING) }.check("cuStreamCreate")?;
        Ok(Arc::new(Self {
            api,
            context,
            raw,
            id: NEXT_LANE.fetch_add(1, Ordering::Relaxed),
            submitted: AtomicU64::new(0),
            published: AtomicU64::new(0),
            copies_seen: AtomicU64::new(0),
        }))
    }

    pub(crate) fn raw(&self) -> CUstream {
        self.raw
    }

    /// Count a submission no token covers yet; called before the launch so
    /// a concurrent scoped wait can never miss it.
    pub(crate) fn mark_unpublished(&self) {
        self.submitted.fetch_add(1, Ordering::AcqRel);
    }

    /// The submissions made so far: what a token minted now covers.
    pub(crate) fn seq(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }

    /// A token covering the first `seq` submissions was recorded on its
    /// storages; later submissions stay unpublished.
    pub(crate) fn publish(&self, seq: u64) {
        self.published.fetch_max(seq, Ordering::AcqRel);
    }

    pub(crate) fn has_unpublished(&self) -> bool {
        self.seq() != self.published.load(Ordering::Acquire)
    }

    /// Count every submission so far as published, returning whether any
    /// was not. Done *before* the corresponding drain: a submission racing
    /// it either precedes the drain or stays unpublished.
    pub(crate) fn take_unpublished(&self) -> bool {
        let seq = self.seq();
        self.published.fetch_max(seq, Ordering::AcqRel) < seq
    }
}

impl Drop for Lane {
    fn drop(&mut self) {
        // SAFETY: the stream this lane created, destroyed in its context;
        // the driver defers destruction until its work retires.
        let result = unsafe {
            (self.api.ctx_set_current)(self.context)
                .check("cuCtxSetCurrent")
                .and_then(|()| (self.api.stream_destroy)(self.raw).check("cuStreamDestroy"))
        };
        if let Err(error) = result {
            tracing::warn!(?error, lane = self.id, "CUDA stream leaked");
        }
    }
}

pub struct CudaDevice {
    api: &'static Api,
    device_id: usize,
    name: String,
    arch: CudaArch,
    limits: CudaLimits,
    /// Host ↔ device copies and memsets (`CudaAllocator`).
    copy: Arc<Lane>,
    /// Per-call `Program::execute` dispatches.
    dispatch: Arc<Lane>,
    /// Every lane built on this device (weak: lanes die with their owners).
    lanes: Mutex<Vec<Weak<Lane>>>,
    /// Storage base → in-flight tokens (see the module docs). Tokens hold the
    /// device; the cycle is moot because opened devices live in the
    /// process-global cache.
    producers: Mutex<HashMap<u64, Producers>>,
    /// Serializes copy-lane enqueue + publication so tokens of the copy lane
    /// are recorded in stream order.
    copy_lock: Mutex<()>,
    /// Newest copy-lane event and its sequence number (0 = none yet).
    copy_tail: Mutex<Option<Arc<CudaEvent>>>,
    copy_seq: AtomicU64,
    /// Recorded at open: the zero of every GPU-clock timestamp.
    base_event: CUevent,
    staging: Mutex<Option<NonNull<u8>>>,
    poisoned: AtomicBool,
    poison_message: OnceLock<String>,
    /// Declared last: released after the lanes destroyed their streams.
    context: PrimaryContext,
}

// SAFETY: every field is either immutable after open or guarded; the staging
// pointer is pinned host memory used only under its mutex.
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDevice")
            .field("device_id", &self.device_id)
            .field("name", &self.name)
            .field("arch", &self.arch)
            .field("poisoned", &self.is_poisoned())
            .finish_non_exhaustive()
    }
}

impl CudaDevice {
    /// Process-global cached open of CUDA device `device_id`. Never panics:
    /// `DeviceUnavailable` without a driver, `NoCudaGpu` without a device,
    /// `CudaDriver` for driver failures.
    pub fn open(device_id: usize) -> Result<Arc<Self>> {
        let mut cache = DEVICE_CACHE.lock();
        if let Some(device) = cache.get(&device_id) {
            return Ok(Arc::clone(device));
        }
        let device = Arc::new(Self::open_uncached(device_id)?);
        cache.insert(device_id, Arc::clone(&device));
        Ok(device)
    }

    /// A private handle on the same primary context, outside the cache; for
    /// tests that poison a device without affecting the shared one.
    pub(crate) fn open_uncached(device_id: usize) -> Result<Self> {
        let api = api()?;
        api.init().map_err(|error| Error::NoCudaGpu { reason: error.to_string() })?;
        let count = api.device_count()?;
        if device_id >= count {
            return Err(Error::NoCudaGpu {
                reason: format!("device_id {device_id} out of range; {count} device(s) present"),
            });
        }
        let mut handle: CUdevice = 0;
        // SAFETY: out-pointer to a live integer; the ordinal was range-checked.
        unsafe { (api.device_get)(&mut handle, device_id as c_int) }.check("cuDeviceGet")?;
        let mut raw = CUcontext::NULL;
        // SAFETY: out-pointer to a live handle slot.
        unsafe { (api.device_primary_ctx_retain)(&mut raw, handle) }.check("cuDevicePrimaryCtxRetain")?;
        // Everything below runs in the retained context; dropping the guard
        // on failure releases it.
        let context = PrimaryContext { api, handle, raw };
        // SAFETY: a context this process retains.
        unsafe { (api.ctx_set_current)(raw) }.check("cuCtxSetCurrent")?;

        let attribute = |id: i32| -> Result<u32> {
            let mut value: c_int = 0;
            // SAFETY: out-pointer to a live integer.
            unsafe { (api.device_get_attribute)(&mut value, id, handle) }.check("cuDeviceGetAttribute")?;
            Ok(u32::try_from(value).unwrap_or(0))
        };
        let mut name = [0 as c_char; 256];
        // SAFETY: the driver writes at most `len` bytes including the NUL.
        unsafe { (api.device_get_name)(name.as_mut_ptr(), name.len() as c_int, handle) }.check("cuDeviceGetName")?;
        // SAFETY: NUL-terminated by the driver (the buffer is zeroed anyway).
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
        let major = attribute(attribute::COMPUTE_CAPABILITY_MAJOR)?;
        let minor = attribute(attribute::COMPUTE_CAPABILITY_MINOR)?;
        let arch = CudaArch::from_compute_capability(
            u8::try_from(major).unwrap_or(u8::MAX),
            u8::try_from(minor).unwrap_or(u8::MAX),
        );
        let limits = CudaLimits {
            sm_count: attribute(attribute::MULTIPROCESSOR_COUNT)?,
            max_threads_per_block: attribute(attribute::MAX_THREADS_PER_BLOCK)?,
            max_threads_per_sm: attribute(attribute::MAX_THREADS_PER_MULTIPROCESSOR)?,
            shared_per_block: attribute(attribute::MAX_SHARED_MEMORY_PER_BLOCK)?,
            warp_size: attribute(attribute::WARP_SIZE)?,
            managed_memory: attribute(attribute::MANAGED_MEMORY)? == 1
                && attribute(attribute::CONCURRENT_MANAGED_ACCESS)? == 1,
        };

        let copy = Lane::create(api, raw)?;
        let dispatch = Lane::create(api, raw)?;
        let mut base_event = CUevent::NULL;
        // SAFETY: out-pointer to a live handle slot; the event is then
        // recorded on the legacy default stream and waited for, so it is
        // complete (and timestamped) before the device is handed out.
        unsafe {
            (api.event_create)(&mut base_event, CU_EVENT_DEFAULT).check("cuEventCreate")?;
            (api.event_record)(base_event, CUstream::NULL).check("cuEventRecord")?;
            (api.event_synchronize)(base_event).check("cuEventSynchronize")?;
        }
        let (driver_major, driver_minor) = api.driver_version()?;
        tracing::info!(
            device_id,
            name,
            %arch,
            sms = limits.sm_count,
            managed = limits.managed_memory,
            driver = format!("{driver_major}.{driver_minor}"),
            scoped_sync = Self::scoped_sync_enabled(),
            "opened CUDA device"
        );
        Ok(Self {
            api,
            device_id,
            name,
            arch,
            limits,
            lanes: Mutex::new(vec![Arc::downgrade(&dispatch)]),
            copy,
            dispatch,
            producers: Mutex::new(HashMap::new()),
            copy_lock: Mutex::new(()),
            copy_tail: Mutex::new(None),
            copy_seq: AtomicU64::new(0),
            base_event,
            staging: Mutex::new(None),
            poisoned: AtomicBool::new(false),
            poison_message: OnceLock::new(),
            context,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn arch(&self) -> CudaArch {
        self.arch
    }

    pub fn limits(&self) -> &CudaLimits {
        &self.limits
    }

    pub(crate) fn api(&self) -> &'static Api {
        self.api
    }

    pub(crate) fn copy_stream(&self) -> CUstream {
        self.copy.raw
    }

    pub(crate) fn dispatch_lane(&self) -> &Arc<Lane> {
        &self.dispatch
    }

    pub(crate) fn base_event(&self) -> CUevent {
        self.base_event
    }

    /// A fresh non-blocking lane, registered for scoped waits.
    pub(crate) fn new_lane(&self) -> Result<Arc<Lane>> {
        self.enter()?;
        let lane = Lane::create(self.api, self.context.raw)?;
        let mut lanes = self.lanes.lock();
        lanes.retain(|weak| weak.strong_count() > 0);
        lanes.push(Arc::downgrade(&lane));
        Ok(lane)
    }

    /// Make this device's context current on the calling thread (the driver
    /// keeps it per thread) and refuse to proceed on a poisoned device. Every
    /// entry point of the backend starts here.
    pub(crate) fn enter(&self) -> Result<&'static Api> {
        if let Some(error) = self.poison_error() {
            return Err(error);
        }
        // SAFETY: a context this device retains for its whole lifetime.
        unsafe { (self.api.ctx_set_current)(self.context.raw) }.check("cuCtxSetCurrent")?;
        Ok(self.api)
    }

    /// [`CUresult::check`] that also latches sticky (context-killing) errors.
    pub(crate) fn check(&self, result: CUresult, call: &'static str) -> Result<()> {
        let outcome = result.check(call);
        if let Err(error) = &outcome
            && result.is_sticky()
        {
            self.poison(&error.to_string());
        }
        outcome
    }

    /// Wait for every stream of this context. Lanes drained here have no
    /// unpublished work left.
    pub fn synchronize(&self) -> Result<()> {
        let api = self.enter()?;
        for lane in self.live_lanes() {
            lane.take_unpublished();
        }
        // SAFETY: plain call in the current context.
        self.check(unsafe { (api.ctx_synchronize)() }, "cuCtxSynchronize")
    }

    pub fn stream_synchronize(&self, stream: CUstream) -> Result<()> {
        let api = self.enter()?;
        // SAFETY: a live stream of this context.
        self.check(unsafe { (api.stream_synchronize)(stream) }, "cuStreamSynchronize")
    }

    /// Drain one lane; its work counts as published from here on.
    pub(crate) fn synchronize_lane(&self, lane: &Lane) -> Result<()> {
        lane.take_unpublished();
        self.stream_synchronize(lane.raw)
    }

    /// `(free, total)` bytes of device memory.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        let api = self.enter()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: out-pointers to live size_t values.
        unsafe { (api.mem_get_info)(&mut free, &mut total) }.check("cuMemGetInfo")?;
        Ok((free, total))
    }

    /// Zero `size` bytes of the storage at `device_ptr` on the copy lane,
    /// ordered after its in-flight producers and before every later launch;
    /// no host wait.
    pub(crate) fn zero(self: &Arc<Self>, device_ptr: CUdeviceptr, size: usize) -> Result<()> {
        self.with_copy_lane(|dev| {
            dev.order_copies_after(&[device_ptr])?;
            // SAFETY: the caller owns `size` bytes at `device_ptr`.
            dev.check(unsafe { (dev.api.memset_d8_async)(device_ptr, 0, size, dev.copy.raw) }, "cuMemsetD8Async")?;
            dev.record_copy(&[device_ptr])
        })
    }

    /// Run one copy-lane operation — `order_copies_after`, the enqueue,
    /// `record_copy` — serialized against the others so the lane's tokens
    /// are published in stream order.
    pub(crate) fn with_copy_lane<T>(self: &Arc<Self>, f: impl FnOnce(&Arc<Self>) -> Result<T>) -> Result<T> {
        let _copy = self.copy_lock.lock();
        f(self)
    }

    /// Run `f` with the pinned bounce buffer (allocated on first use). The
    /// lock serializes staged copies, which share the copy stream anyway.
    pub(crate) fn with_staging<T>(&self, f: impl FnOnce(NonNull<u8>, usize) -> Result<T>) -> Result<T> {
        let mut staging = self.staging.lock();
        let pointer = match *staging {
            Some(pointer) => pointer,
            None => {
                let api = self.enter()?;
                let mut raw = null_mut();
                // SAFETY: out-pointer to a live pointer slot.
                unsafe { (api.mem_host_alloc)(&mut raw, STAGING_BYTES, CU_MEMHOSTALLOC_PORTABLE) }
                    .check("cuMemHostAlloc")?;
                let pointer = NonNull::new(raw.cast::<u8>())
                    .ok_or_else(|| Error::Runtime { message: "cuMemHostAlloc returned null".into() })?;
                *staging = Some(pointer);
                pointer
            }
        };
        f(pointer, STAGING_BYTES)
    }

    /// `true` once a sticky driver error has poisoned the context.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Latch a fatal error: the context is unusable, the message is kept.
    pub fn poison(&self, message: &str) {
        let _ = self.poison_message.set(message.to_string());
        self.poisoned.store(true, Ordering::Release);
    }

    pub fn poison_error(&self) -> Option<Error> {
        self.is_poisoned().then(|| Error::Runtime {
            message: self.poison_message.get().cloned().unwrap_or_else(|| "CUDA device poisoned".into()),
        })
    }
}

/// Scoped synchronization (see the module docs).
impl CudaDevice {
    /// Kill switch: `SVOD_CUDA_SCOPED_SYNC=0` makes every wait a context
    /// drain and every copy synchronous, for bisecting scoped-sync regressions.
    pub(crate) fn scoped_sync_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("SVOD_CUDA_SCOPED_SYNC").as_deref() != Ok("0"))
    }

    /// Pre-register a storage so "known storage, nothing in flight" is
    /// distinguishable from "unknown storage" (conservative drain).
    pub(crate) fn register_storage(&self, base: u64) {
        if Self::scoped_sync_enabled() {
            self.producers.lock().entry(base).or_default();
        }
    }

    pub(crate) fn unregister_storage(&self, base: u64) {
        self.producers.lock().remove(&base);
    }

    /// Record `token` as an in-flight producer/reader of the storage at
    /// `base`, replacing this lane's previous token. A token of another
    /// backend cannot be ordered by event, so the storage's producers become
    /// unknown (every later access drains).
    pub(crate) fn record_producer(&self, base: u64, token: &Arc<dyn CompletionToken>) {
        if !Self::scoped_sync_enabled() {
            return;
        }
        let Some(token) = (token.as_ref() as &dyn Any).downcast_ref::<CudaCompletionToken>() else {
            tracing::debug!(base, "non-CUDA completion token recorded on a CUDA storage; producers unknown");
            self.producers.lock().remove(&base);
            return;
        };
        self.record_cuda_producer(base, token);
    }

    fn record_cuda_producer(&self, base: u64, token: &CudaCompletionToken) {
        let mut producers = self.producers.lock();
        let tokens = producers.entry(base).or_default();
        match tokens.iter_mut().find(|earlier| earlier.lane() == token.lane()) {
            Some(slot) => *slot = token.clone(),
            None => tokens.push(token.clone()),
        }
        if tokens.len() > PRUNE_PRODUCERS_ABOVE {
            tokens.retain(|token| !token.retired());
        }
    }

    #[cfg(test)]
    pub(crate) fn producer_count(&self, base: u64) -> Option<usize> {
        self.producers.lock().get(&base).map(|tokens| tokens.len())
    }

    fn live_lanes(&self) -> Vec<Arc<Lane>> {
        let mut lanes = self.lanes.lock();
        lanes.retain(|weak| weak.strong_count() > 0);
        lanes.iter().filter_map(Weak::upgrade).collect()
    }

    /// Wait on the host for everything that may still touch the storage at
    /// `base`: its recorded tokens and every lane with unpublished
    /// submissions. Unknown storages (and the kill switch) drain the context.
    pub(crate) fn wait_storage(&self, base: u64) -> Result<()> {
        if !Self::scoped_sync_enabled() {
            return self.synchronize();
        }
        self.enter()?;
        // The guard must not outlive the lookup: the drain below can take a
        // kernel's duration and every publication needs the table.
        let tokens = self.producers.lock().get(&base).cloned();
        let Some(tokens) = tokens else { return self.synchronize() };
        for lane in self.live_lanes() {
            if lane.take_unpublished() {
                self.stream_synchronize(lane.raw)?;
            }
        }
        for token in &tokens {
            token.event().wait(0).inspect_err(|error| self.poison(&error.to_string()))?;
        }
        if let Some(current) = self.producers.lock().get_mut(&base) {
            current.retain(|token| !tokens.iter().any(|waited| Arc::ptr_eq(waited.event(), token.event())));
        }
        Ok(())
    }

    /// Order the copy lane after everything that may still touch `bases`,
    /// on the GPU (`cuStreamWaitEvent`), so the following copy needs no host
    /// wait. Lanes with unpublished submissions contribute a tail event;
    /// unknown storages drain the context. Call within `with_copy_lane`.
    pub(crate) fn order_copies_after(self: &Arc<Self>, bases: &[u64]) -> Result<()> {
        if !Self::scoped_sync_enabled() {
            return self.synchronize();
        }
        let api = self.enter()?;
        let mut events: Vec<Arc<CudaEvent>> = Vec::new();
        {
            let producers = self.producers.lock();
            for base in bases {
                let Some(tokens) = producers.get(base) else {
                    drop(producers);
                    return self.synchronize();
                };
                events.extend(tokens.iter().map(|token| Arc::clone(token.event())));
            }
        }
        for lane in self.live_lanes() {
            if lane.has_unpublished() {
                let tail = CudaEvent::new(Arc::clone(self), false)?;
                tail.record(lane.raw)?;
                events.push(Arc::new(tail));
            }
        }
        for event in events.iter().filter(|event| !event.observed_complete()) {
            // SAFETY: live stream and event of this context.
            self.check(unsafe { (api.stream_wait_event)(self.copy.raw, event.raw, 0) }, "cuStreamWaitEvent")?;
        }
        Ok(())
    }

    /// Publish the copy-lane work just enqueued as the newest producer of
    /// `bases` and as the copy tail every later launch waits on. Call within
    /// `with_copy_lane`, after the enqueue.
    pub(crate) fn record_copy(self: &Arc<Self>, bases: &[u64]) -> Result<()> {
        if !Self::scoped_sync_enabled() {
            return self.stream_synchronize(self.copy.raw);
        }
        let event = CudaEvent::new(Arc::clone(self), false)?;
        event.record(self.copy.raw)?;
        let event = Arc::new(event);
        let token = CudaCompletionToken::new(Arc::clone(&event), self.copy.id);
        for base in bases {
            self.record_cuda_producer(*base, &token);
        }
        *self.copy_tail.lock() = Some(event);
        self.copy_seq.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// Order a launch on `lane` after every copy-lane operation published so
    /// far; a no-op unless a copy happened since the lane's last launch.
    pub(crate) fn order_launch(&self, lane: &Lane) -> Result<()> {
        let seq = self.copy_seq.load(Ordering::Acquire);
        if seq == lane.copies_seen.load(Ordering::Acquire) {
            return Ok(());
        }
        let tail = self.copy_tail.lock().clone();
        if let Some(event) = tail.filter(|event| !event.observed_complete()) {
            let api = self.enter()?;
            // SAFETY: live stream and event of this context.
            self.check(unsafe { (api.stream_wait_event)(lane.raw, event.raw, 0) }, "cuStreamWaitEvent")?;
        }
        lane.copies_seen.store(seq, Ordering::Release);
        Ok(())
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        let api = self.api;
        // SAFETY: handles this device created; the lanes and the context
        // release themselves afterwards, in field order.
        unsafe {
            if (api.ctx_set_current)(self.context.raw) != CUresult::SUCCESS {
                return;
            }
            if let Some(staging) = self.staging.get_mut().take() {
                (api.mem_free_host)(staging.as_ptr().cast());
            }
            (api.event_destroy)(self.base_event);
        }
    }
}

/// The retained primary context, released on drop.
struct PrimaryContext {
    api: &'static Api,
    handle: CUdevice,
    raw: CUcontext,
}

impl Drop for PrimaryContext {
    fn drop(&mut self) {
        // SAFETY: balances the retain that created this value.
        unsafe { (self.api.device_primary_ctx_release)(self.handle) };
    }
}

/// An owned lane of a device.
pub struct CudaStream {
    dev: Arc<CudaDevice>,
    lane: Arc<Lane>,
}

impl CudaStream {
    /// A non-blocking stream (not ordered against the legacy default stream).
    pub fn new(dev: Arc<CudaDevice>) -> Result<Self> {
        let lane = dev.new_lane()?;
        Ok(Self { dev, lane })
    }

    pub fn raw(&self) -> CUstream {
        self.lane.raw
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.dev
    }

    pub(crate) fn lane(&self) -> &Arc<Lane> {
        &self.lane
    }

    pub fn synchronize(&self) -> Result<()> {
        self.dev.synchronize_lane(&self.lane)
    }

    /// Record a fresh event at the current tail of this stream.
    pub fn record(&self, timing: bool) -> Result<Arc<CudaEvent>> {
        let event = CudaEvent::new(Arc::clone(&self.dev), timing)?;
        event.record(self.lane.raw)?;
        Ok(Arc::new(event))
    }

    /// A completion token for everything submitted so far. The submissions
    /// stay unpublished until the owner recorded the token on its storages
    /// and called [`CompletionToken::published`].
    pub fn token(&self) -> Result<CudaCompletionToken> {
        let seq = self.lane.seq();
        Ok(CudaCompletionToken::new(self.record(false)?, self.lane.id).covering(&self.lane, seq))
    }
}

/// An owned event of a device.
pub struct CudaEvent {
    dev: Arc<CudaDevice>,
    raw: CUevent,
    /// Completion, once observed, is final until the next record.
    done: AtomicBool,
}

impl CudaEvent {
    /// `timing` events carry GPU timestamps (`cuEventElapsedTime`);
    /// completion-only events skip them and are cheaper to record.
    pub fn new(dev: Arc<CudaDevice>, timing: bool) -> Result<Self> {
        let api = dev.enter()?;
        let mut raw = CUevent::NULL;
        let flags = if timing { CU_EVENT_DEFAULT } else { CU_EVENT_DISABLE_TIMING };
        // SAFETY: out-pointer to a live handle slot.
        unsafe { (api.event_create)(&mut raw, flags) }.check("cuEventCreate")?;
        Ok(Self { dev, raw, done: AtomicBool::new(false) })
    }

    pub fn raw(&self) -> CUevent {
        self.raw
    }

    pub fn record(&self, stream: CUstream) -> Result<()> {
        let api = self.dev.enter()?;
        self.done.store(false, Ordering::Release);
        // SAFETY: live event and stream of this context.
        self.dev.check(unsafe { (api.event_record)(self.raw, stream) }, "cuEventRecord")
    }

    /// Whether completion was already observed (no driver call).
    pub(crate) fn observed_complete(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    /// Whether the recorded work has completed (`cuEventQuery`). An event
    /// never recorded counts as completed, as the driver defines it.
    pub fn completed(&self) -> Result<bool> {
        if self.observed_complete() {
            return Ok(true);
        }
        let api = self.dev.enter()?;
        // SAFETY: a live event.
        match unsafe { (api.event_query)(self.raw) } {
            CUresult::SUCCESS => {
                self.done.store(true, Ordering::Release);
                Ok(true)
            }
            CUresult::NOT_READY => Ok(false),
            other => self.dev.check(other, "cuEventQuery").map(|()| true),
        }
    }

    /// Block until completion; `timeout_ms == 0` waits forever.
    pub fn wait(&self, timeout_ms: u64) -> Result<()> {
        if self.observed_complete() {
            return Ok(());
        }
        if timeout_ms == 0 {
            let api = self.dev.enter()?;
            // SAFETY: a live event.
            self.dev.check(unsafe { (api.event_synchronize)(self.raw) }, "cuEventSynchronize")?;
            self.done.store(true, Ordering::Release);
            return Ok(());
        }
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        while !self.completed()? {
            if Instant::now() >= deadline {
                return TimelineTimeoutSnafu { what: "CUDA event", target: 1u64, current: 0u64, waited_ms: timeout_ms }
                    .fail();
            }
            std::thread::sleep(EVENT_POLL);
        }
        Ok(())
    }

    /// Milliseconds on the GPU clock from `start` to `self`; both must be
    /// completed timing events.
    pub fn elapsed_ms_since(&self, start: CUevent) -> Result<f32> {
        let api = self.dev.enter()?;
        let mut ms = 0f32;
        // SAFETY: out-pointer to a live float; both events are live.
        self.dev.check(unsafe { (api.event_elapsed_time)(&mut ms, start, self.raw) }, "cuEventElapsedTime")?;
        Ok(ms)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let api = self.dev.api();
        // SAFETY: an event this value created, destroyed in its context; the
        // driver defers destruction until it completes.
        let result = unsafe {
            (api.ctx_set_current)(self.dev.context.raw)
                .check("cuCtxSetCurrent")
                .and_then(|()| (api.event_destroy)(self.raw).check("cuEventDestroy"))
        };
        if let Err(error) = result {
            tracing::warn!(?error, "CUDA event leaked");
        }
    }
}

impl std::fmt::Debug for CudaEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaEvent({:p})", self.raw.0)
    }
}
