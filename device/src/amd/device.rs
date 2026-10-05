//! `AmdDevice`: KFD-direct device handle.
//!
//! Opens `/dev/kfd` and `/dev/dri/renderD*`, parses topology, calls
//! `AMDKFD_IOC_ACQUIRE_VM`. Owns an `Arc<AmdDeviceCore>` (the immutable
//! per-physical-AMD:N identity — KFD/DRM fds, topology, event-page state,
//! poison latch, shared signal pool, and bounded exclusive-lane pool). It holds
//! no execution context of its own: logical owners acquire non-clone queue
//! leases for publication, and the device-wide synchronize
//! chain (`AmdAllocator::_copyin`/`_copyout`/`_free`) drains every registered
//! pool queue via `synchronize_all`.

#![cfg(unix)]

use std::collections::HashMap;
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use nix::fcntl::{OFlag, open};
use nix::sys::stat::Mode;
use parking_lot::Mutex;
use std::sync::LazyLock;
use svod_dtype::AmdArch;
use tracing::debug;

use crate::amd::sys::{ioctl, kfd};
use crate::amd::topology::{AmdNode, enumerate};
use crate::error::{Error, Result};

/// Per-process cache of opened `AmdDevice`s, keyed by `device_id`. KFD only
/// accepts one `ACQUIRE_VM` per (process, GPU); the cache ensures that
/// concurrent `AmdAllocator::new(0)` calls — e.g. registry-cached
/// LRU-wrapped allocator + factory-side queue/arena setup — share the same
/// `Arc<AmdDevice>` instead of double-opening.
///
/// The cached `Arc<AmdDevice>` carries the shared `Arc<AmdDeviceCore>` —
/// per-plan and per-graph callers reach the core via `AmdDevice::core()` and
/// are assigned a shared `PoolQueue` against it (no extra KFD opens).
static DEVICE_CACHE: LazyLock<Mutex<HashMap<usize, Arc<AmdDevice>>>> = LazyLock::new(Default::default);

/// Process-wide `/dev/kfd` handle. KFD is opened once per process and all
/// devices share it — events created on a per-device basis are addressed by
/// `event_id` against this shared fd.
static GLOBAL_KFD: LazyLock<Mutex<Option<Arc<OwnedFd>>>> = LazyLock::new(Default::default);

/// Process-wide event-page state. The 0x8000 GTT event page is allocated
/// exactly once per process; the first device
/// allocates+binds it via `CREATE_EVENT(event_page_offset=handle)`,
/// subsequent devices just `MAP_MEMORY_TO_GPU` it to their `gpu_id`.
static EVENT_PAGE: LazyLock<Mutex<Option<EventPageState>>> = LazyLock::new(Default::default);

#[derive(Debug, Clone, Copy)]
pub(crate) struct EventPageState {
    pub(crate) handle: u64,
    pub(crate) va: u64,
    pub(crate) size: usize,
}

struct EventPageAllocation<'a> {
    kfd_fd: &'a OwnedFd,
    state: EventPageState,
    gpu_id: u32,
    gpu_mapped: bool,
    committed: bool,
}

impl EventPageAllocation<'_> {
    fn commit(mut self) -> EventPageState {
        self.committed = true;
        self.state
    }
}

impl Drop for EventPageAllocation<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if self.gpu_mapped {
            let mut gpu_id = self.gpu_id;
            let mut args = kfd::kfd_ioctl_unmap_memory_from_gpu_args {
                handle: self.state.handle,
                device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
                n_devices: 1,
                n_success: 0,
            };
            let _ = unsafe { ioctl::kfd_unmap_memory_from_gpu(self.kfd_fd.as_raw_fd(), &mut args as *mut _) };
        }
        unsafe { libc::munmap(self.state.va as *mut _, self.state.size) };
        let mut args = kfd::kfd_ioctl_free_memory_of_gpu_args { handle: self.state.handle };
        let _ = unsafe { ioctl::kfd_free_memory_of_gpu(self.kfd_fd.as_raw_fd(), &mut args as *mut _) };
    }
}

/// Scratch backing memory + `COMPUTE_TMPRING_SIZE` packing. Held under a
/// mutex on `PoolQueue` so [`PoolQueue::ensure_has_local_memory`](crate::amd::connector::PoolQueue::ensure_has_local_memory)
/// can grow the scratch buffer when a freshly-loaded program demands more
/// bytes per thread than what's currently allocated. `pub(crate)` because the
/// owning field lives in the sibling `connector` module.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScratchState {
    /// GPU VA of the current scratch buffer.
    pub gpu_va: u64,
    /// Bytes per thread (rounded up to 4-byte slot stride for wave64).
    /// Equivalent to the kernel's `max_private_segment_size`.
    pub size_per_thread: u32,
    /// Packed `COMPUTE_TMPRING_SIZE` register value.
    pub tmpring_size: u32,
    /// KFD handle + total byte size of the backing alloc — needed to free the
    /// old buffer when scratch grows.
    pub handle: u64,
    pub size: usize,
}

/// The private-segment (scratch) fields the AQL packet processor reads from the
/// `amd_queue_t` GART descriptor. On the PM4 dispatch path the same information
/// is pushed via `COMPUTE_TMPRING_SIZE` / `COMPUTE_DISPATCH_SCRATCH_BASE` and
/// the user-SGPR descriptor instead; every AQL queue — multi-XCC CDNA and any
/// forced-AQL gfx11+ queue — consumes it from the descriptor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct AqlScratchDesc {
    /// `amd_queue_t.scratch_backing_memory_location` — scratch buffer VA.
    pub backing_va: u64,
    /// `amd_queue_t.compute_tmpring_size`.
    pub tmpring_size: u32,
    /// `amd_queue_t.scratch_resource_descriptor` — a 4-dword SQ_BUF_RSRC.
    pub resource_descriptor: [u32; 4],
    /// `amd_queue_t.scratch_wave64_lane_byte_size` — per-lane byte size (wave64,
    /// so just the per-thread private segment size).
    pub wave64_lane_byte_size: u32,
}

/// gfx9 SQ_BUF_RSRC WORD3 for a scratch buffer: DST_SEL=XYZW (4,5,6,7),
/// NUM_FORMAT=UINT(4), DATA_FORMAT=32(4), ELEMENT_SIZE=1, INDEX_STRIDE=3,
/// ADD_TID_ENABLE=1, TYPE=SQ_RSRC_BUF(0).
const SCRATCH_RSRC_WORD3_GFX9: u32 = 0x00EA_4FAC;

/// gfx11 SQ_BUF_RSRC WORD3 for a scratch buffer (ROCr
/// `FillBufRsrcWord3_Gfx11`): DST_SEL=XYZW (4,5,6,7), FORMAT=32_UINT (0x14),
/// INDEX_STRIDE=0 (filled in by the CP), ADD_TID_ENABLE=1, OOB_SELECT=2 (no
/// bounds check in swizzle mode), TYPE=SQ_RSRC_BUF(0). The gfx12 layout only
/// adds compression bits that stay zero, so the same value serves both.
const SCRATCH_RSRC_WORD3_GFX11: u32 = 0x2081_4FAC;

impl AqlScratchDesc {
    /// Build the gfx9 scratch descriptor. `size_per_xcc` goes in NUM_RECORDS
    /// (WORD2) — the per-XCC slice of the shared backing buffer — and the
    /// SWIZZLE_ENABLE bit (WORD1 bit 31) enables the per-thread scratch swizzle.
    pub(crate) fn gfx9(scratch_va: u64, size_per_xcc: usize, tmpring_size: u32, private_segment_size: u32) -> Self {
        Self {
            backing_va: scratch_va,
            tmpring_size,
            resource_descriptor: [
                scratch_va as u32,
                ((scratch_va >> 32) as u32 & 0xFFFF) | 0x8000_0000,
                size_per_xcc as u32,
                SCRATCH_RSRC_WORD3_GFX9,
            ],
            wave64_lane_byte_size: private_segment_size,
        }
    }

    /// Build the gfx11/gfx12 scratch descriptor (ROCr `InitScratchSRD` case
    /// 11): SWIZZLE_ENABLE moves to WORD1 bits [31:30] (value 1), NUM_RECORDS
    /// carries the full backing size (one XCC on RDNA), and
    /// `scratch_wave64_lane_byte_size` records the rounded per-thread stride.
    /// Consumed by any gfx11+ AQL queue (`SVOD_AMD_AQL=1`), never by the PM4
    /// path, which programs scratch via per-dispatch `SET_SH_REG`.
    pub(crate) fn gfx11(scratch_va: u64, size: usize, tmpring_size: u32, size_per_thread: u32) -> Self {
        Self {
            backing_va: scratch_va,
            tmpring_size,
            resource_descriptor: [
                scratch_va as u32,
                ((scratch_va >> 32) as u32 & 0xFFFF) | 0x4000_0000,
                size as u32,
                SCRATCH_RSRC_WORD3_GFX11,
            ],
            wave64_lane_byte_size: size_per_thread,
        }
    }
}

/// Immutable per-physical-AMD:N identity: KFD/DRM fds, topology, event-page
/// state, fault latch, shared signal pool, the bounded shared-queue pool. One
/// instance per physical GPU (KFD rejects double `ACQUIRE_VM`); shared as
/// `Arc<AmdDeviceCore>` by every `PoolQueue` built against the device. The
/// queue registry and `synchronize_all` live here so the host can drain every
/// live queue before any destructive operation (`AmdAllocator::_free`, etc.).
///
/// The backend seam: all KFD ioctls (memory alloc/free, queue ring setup/
/// teardown, event waits) route through `iface`. KFD-specific state (the kfd/
/// drm fds, ABI version, event ids, event page) lives on the [`KfdIface`]
/// implementor, not the core. `node` + `arch` stay here as the device identity.
/// In-flight completion tokens for one storage (usually 0-2 concurrent
/// plans touch a given storage).
type StorageTokens = smallvec::SmallVec<[Arc<dyn crate::sync::CompletionToken>; 2]>;

pub struct AmdDeviceCore {
    pub node: AmdNode,
    pub arch: AmdArch,
    /// Backend implementation (KFD today). All ioctls route through this.
    iface: Arc<dyn crate::amd::iface::AmdIface>,
    /// Whether an SDMA copy queue is available on this physical device. Set
    /// by the factory once it installs one (every AMD part unless
    /// `AMD_DISABLE_SDMA` is set or creation fails); while `false`, every
    /// buffer is forced host-visible.
    has_sdma_queue: AtomicBool,
    /// Opt-in for PM4 single-XCC graph capture (see `AmdGraph::capture`). Default
    /// `false` (per-call dispatch) — capture is a measured regression on gfx1151
    /// despite a bit-identical transcript, so it's exposed only for hardware that
    /// benefits or future barrier-granularity work. Set by the device factory;
    /// a per-device flag (not an env var) so tests toggle it without a process-
    /// global env race and `=0` can't accidentally enable it.
    pm4_graph: AtomicBool,
    /// Poison latch. Once a GPU fault/timeout is observed, the device is dead:
    /// every `synchronize`/`execute` against any connector on this device
    /// fails fast. Per-physical
    /// device because a memory fault corrupts the whole VM, not just one queue.
    poisoned: AtomicBool,
    error_msg: OnceLock<String>,
    /// Registry of every pool queue built against this core. Weak so dropped
    /// queues don't stay alive. Used by [`AmdDeviceCore::synchronize_all`] to
    /// drain ALL in-flight GPU work before destructive host-visible operations
    /// (`AmdAllocator::_copyin`/`_copyout`/`_free`). The drain
    /// (`PoolQueue::drain_all`) reads only timeline signal slots and does not
    /// take publication locks.
    pub(crate) connectors: parking_lot::Mutex<Vec<Weak<crate::amd::connector::PoolQueue>>>,
    /// Scoped-sync producer table: storage base VA → completion tokens of the
    /// submissions that may still touch it (readers AND writers — a host
    /// overwrite is a WAR hazard against in-flight readers too). Retired
    /// tokens are pruned on insert and wait. A VA absent from the table falls
    /// back to `synchronize_all` (unknown producer → conservative drain).
    producers: parking_lot::Mutex<std::collections::HashMap<u64, StorageTokens>>,
    /// Submissions with no durable owner (`Program::execute` fire-and-forget
    /// fallback, e.g. BEAM timing) — waited by every scoped wait as a safety
    /// net; normally empty.
    unattributed: parking_lot::Mutex<Vec<Arc<dyn crate::sync::CompletionToken>>>,
    /// The device's single 16 MiB kernarg arena, shared by every lane (tinygrad
    /// has one `kernargs_buf` per device). `Weak` so the arena still dies with
    /// the last `PoolQueue` holding it — the core outlives every queue.
    pub(crate) kernarg_arena: parking_lot::Mutex<Weak<crate::amd::kernarg::KernargArena>>,
    /// Process-global signal pool, allocated once per physical device. Lazily
    /// installed by the device factory and shared across every `PoolQueue`
    /// (PM4 counter signal acquired here at queue construction) — pool access
    /// is rare (slot alloc on queue build), and one pool covers many
    /// queues at 4 KiB total VRAM.
    signal_pool: OnceLock<Arc<crate::amd::signal::SignalPool>>,
    /// Process-global SDMA copy queue, installed by the factory once the signal
    /// pool exists. Its presence is what flips `has_sdma_queue` true and so
    /// enables device-local (non-host-visible) buffers; the allocator's
    /// device-only copy arms route through it.
    copy_queue: OnceLock<Arc<crate::amd::queue::AmdCopyQueue>>,
    /// Bounded pool of lazily-created compute lanes. A non-clone `QueueLease`
    /// atomically claims one initialized lane; contention parks after the pool
    /// reaches its hardware cap rather than co-tenanting a mutable ring.
    queue_pool: crate::amd::connector::QueuePool,
    /// Max distinct KFD compute queues in the pool. Read once at open from
    /// `SVOD_AMD_HW_QUEUES` (default 4 on multi-XCC, else 1; min 1). The per-process hardware budget
    /// is small (~24 user compute queues on CDNA; HIP's `GPU_MAX_HW_QUEUES`
    /// defaults to 4), so acquisition parks after reaching the cap.
    hw_queues: usize,
}

/// Open handle to one AMD GPU node.
///
/// A thin owner of the immutable `AmdDeviceCore`. There is no per-device
/// "default" queue: plans and graphs hold logical contexts while publication
/// acquires an exclusive lane. The
/// device-wide synchronize chain (`AmdAllocator::_copyin`/`_copyout`/`_free`)
/// routes through `dev.synchronize() → core.synchronize_all()`, which drains
/// EVERY pool queue registered on the core.
///
/// Immutable Core fields stay reachable via [`Deref`](std::ops::Deref) — `self.dev.node`,
/// `self.dev.kfd_fd`, `self.dev.poison_error()`, etc.
#[derive(Debug)]
pub struct AmdDevice {
    /// Immutable identity (cloneable across connectors).
    core: Arc<AmdDeviceCore>,
}

impl std::ops::Deref for AmdDevice {
    type Target = AmdDeviceCore;
    #[inline]
    fn deref(&self) -> &AmdDeviceCore {
        &self.core
    }
}

impl AmdDevice {
    /// Build a host-only device around a synthetic topology node and backend.
    /// This deliberately bypasses the device cache, KFD, and process-global
    /// event-page setup so lifecycle tests can exercise the core on any host.
    #[cfg(test)]
    pub(crate) fn synthetic_with_xcc(iface: Arc<dyn crate::amd::iface::AmdIface>, num_xcc: u32) -> Arc<Self> {
        let node = AmdNode {
            node_id: 0,
            gpu_id: 1,
            drm_render_minor: 0,
            gfx_target_version: 110_000,
            simd_count: 4,
            array_count: 1,
            simd_arrays_per_engine: 1,
            simd_per_cu: 4,
            max_waves_per_simd: 8,
            lds_size_in_kb: 64,
            wave_front_size: 32,
            num_xcc,
            num_cp_queues: 1,
            max_slots_scratch_cu: 32,
        };
        let core = Arc::new(AmdDeviceCore {
            node,
            arch: AmdArch::Gfx1100,
            iface,
            has_sdma_queue: AtomicBool::new(false),
            pm4_graph: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            error_msg: OnceLock::new(),
            connectors: parking_lot::Mutex::new(Vec::new()),
            producers: parking_lot::Mutex::new(std::collections::HashMap::new()),
            unattributed: parking_lot::Mutex::new(Vec::new()),
            kernarg_arena: parking_lot::Mutex::new(Weak::new()),
            signal_pool: OnceLock::new(),
            copy_queue: OnceLock::new(),
            queue_pool: crate::amd::connector::QueuePool::new(1),
            hw_queues: 1,
        });
        Arc::new(Self { core })
    }

    /// Open the `device_id`-th GPU node.
    ///
    /// Returns:
    /// - `Err(NoAmdGpu)` when there is no `/dev/kfd`, no GPU nodes in
    ///   topology, or `device_id` is out of range. Never panics.
    /// - `Err(DeviceUnavailable)` when the host has hardware we don't support
    ///   (hardware outside the supported `AmdArch` set).
    /// - `Err(AmdIoctl)` for KFD failures (permission denied, no event page).
    pub fn open(device_id: usize) -> Result<Arc<Self>> {
        // KFD permits one process VM acquisition per GPU. Keep first-open under
        // the cache lock so concurrent callers cannot construct distinct cores.
        let mut cache = DEVICE_CACHE.lock();
        if let Some(dev) = cache.get(&device_id) {
            return Ok(Arc::clone(dev));
        }
        let dev = Self::open_uncached(device_id)?;
        cache.insert(device_id, Arc::clone(&dev));
        Ok(dev)
    }

    fn open_uncached(device_id: usize) -> Result<Arc<Self>> {
        let nodes = enumerate();
        if nodes.is_empty() {
            return Err(Error::NoAmdGpu { reason: "no KFD topology nodes (no /dev/kfd?)".into() });
        }
        let node = nodes
            .get(device_id)
            .ok_or_else(|| Error::NoAmdGpu {
                reason: format!("device_id {device_id} out of range; {} GPU node(s) present", nodes.len()),
            })?
            .clone();
        let arch =
            AmdArch::from_gfx_target_version(node.gfx_target_version).ok_or_else(|| Error::DeviceUnavailable {
                reason: format!(
                    "unsupported gfx target {} (decoded major.minor.step = {}.{}.{}); supported families: \
                 CDNA gfx942/950, RDNA3 gfx1100/1101/1102/1151, RDNA4 gfx1200/1201",
                    node.gfx_target_version,
                    node.gfx_target_version / 10_000,
                    (node.gfx_target_version / 100) % 100,
                    node.gfx_target_version % 100,
                ),
            })?;

        // Backend selection. Today only the KFD-direct backend exists; the
        // `SVOD_AMD_BACKEND` knob is the seam where the userspace AM driver
        // will plug in. All KFD bring-up + ioctls live on `KfdIface`.
        let backend = std::env::var("SVOD_AMD_BACKEND").unwrap_or_else(|_| "kfd".into());
        if backend != "kfd" {
            return Err(Error::NoAmdGpu {
                reason: format!("unknown SVOD_AMD_BACKEND={backend} (only 'kfd' supported)"),
            });
        }
        let iface: Arc<dyn crate::amd::iface::AmdIface> = Arc::new(crate::amd::iface::KfdIface::open(&node)?);

        debug!(node = node.node_id, gpu_id = node.gpu_id, arch = arch.mcpu(), backend = %backend, "AmdDevice opened");

        // Bounded exclusive-lane pool: at most `SVOD_AMD_HW_QUEUES` distinct
        // KFD compute queues. The default is scheduler-aware, mirroring where
        // the ROCm stack itself ships multi-queue:
        //   - multi-XCC CDNA (MI300-class): 4 — queues are HWS/MEC-runlist
        //     scheduled and concurrent AQL queues are the validated production
        //     configuration (HIP's own GPU_MAX_HW_QUEUES default is 4);
        //   - single-XCC gfx11+ (MES-scheduled, RDNA3/3.5/4): 1 — feeding
        //     several PM4 user queues concurrently parks CP micro-engines in
        //     `WAIT_REG_MEM` spins that MES cannot preempt and wedges the
        //     firmware into an unrecoverable reset (reproduced on gfx1151;
        //     see the crate-level AMD notes), and gfx11 MEC firmware implements no
        //     scheduler-visible timeline-wait packet to lower waits onto
        //     (the AMD barrier-value vendor packet is an illegal opcode
        //     there). The env override remains for validation experiments.
        let hw_queues = std::env::var("SVOD_AMD_HW_QUEUES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(if node.num_xcc.max(1) > 1 { 4 } else { 1 })
            .clamp(1, u64::BITS as usize);

        let core = Arc::new(AmdDeviceCore {
            node,
            arch,
            iface,
            has_sdma_queue: AtomicBool::new(false),
            pm4_graph: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            error_msg: OnceLock::new(),
            copy_queue: OnceLock::new(),
            connectors: parking_lot::Mutex::new(Vec::new()),
            producers: parking_lot::Mutex::new(std::collections::HashMap::new()),
            unattributed: parking_lot::Mutex::new(Vec::new()),
            kernarg_arena: parking_lot::Mutex::new(Weak::new()),
            signal_pool: OnceLock::new(),
            queue_pool: crate::amd::connector::QueuePool::new(hw_queues),
            hw_queues,
        });
        Ok(Arc::new(Self { core }))
    }

    /// Borrow the shared immutable core without re-acquiring KFD.
    #[inline]
    pub fn core(&self) -> &Arc<AmdDeviceCore> {
        &self.core
    }

    /// Drain all submitted GPU work on every pool queue backed by this device.
    /// Lanes retain independent queue timelines, so the drain must cover every
    /// registered queue. Skipping one would let
    /// `AmdAllocator::_copyout`/`_copyin`/`_free` observe an unfinished
    /// kernel's buffer.
    pub fn synchronize(&self) -> Result<()> {
        self.core.synchronize_all()
    }
}

impl std::fmt::Debug for AmdDeviceCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdDeviceCore").field("node", &self.node).field("arch", &self.arch).finish_non_exhaustive()
    }
}

impl AmdDeviceCore {
    /// Drain every pool queue backed by this core — the per-VM fence before any
    /// destructive host-visible op (`AmdAllocator::_copyin`/`_copyout`/`_free`).
    /// Iterates the queue registry and drains each via `PoolQueue::drain_all`,
    /// which reads only the PM4 counter atomic and signal slots. A freed/read
    /// buffer has no live handle, so owners cannot add
    /// new work referencing it; draining each queue fences all in-flight
    /// readers. Fast on idle queues.
    pub fn synchronize_all(&self) -> Result<()> {
        if let Some(err) = self.poison_error() {
            return Err(err);
        }
        // Snapshot strong refs to release the registry lock before the
        // potentially multi-second waits, keeping each queue alive across its
        // drain so a concurrent queue drop can't pull the rug out.
        let live: Vec<Arc<crate::amd::connector::PoolQueue>> =
            self.connectors.lock().iter().filter_map(|w| w.upgrade()).collect();
        // Drain every queue timeline and retained linked-plan finalizer; collect the first
        // error but keep going so one stuck queue doesn't strand buffer-frees on
        // the others.
        let mut first_err: Option<Error> = None;
        for q in live {
            if let Err(e) = q.drain_all() {
                tracing::warn!(?e, "synchronize_all: queue drain failed; continuing");
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        // Opportunistic GC of dropped queue entries. The registry is touched
        // here on every host read/free, so dead Weaks don't accumulate.
        self.connectors.lock().retain(|w| w.strong_count() > 0);
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Kill switch for storage-scoped host synchronization
    /// (`SVOD_AMD_SCOPED_SYNC=0` → every wait falls back to
    /// `synchronize_all`), for bisecting scoped-sync regressions.
    fn scoped_sync_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("SVOD_AMD_SCOPED_SYNC").as_deref() != Ok("0"))
    }

    /// Pre-register a storage VA in the producer table so "known storage,
    /// nothing in flight" is distinguishable from "unknown VA" (conservative
    /// full drain).
    pub(crate) fn register_storage(&self, base: u64) {
        if Self::scoped_sync_enabled() {
            self.producers.lock().entry(base).or_default();
        }
    }

    pub(crate) fn unregister_storage(&self, base: u64) {
        self.producers.lock().remove(&base);
    }

    /// Record `token` as an in-flight producer/reader of the storage at `base`.
    pub(crate) fn record_producer(&self, base: u64, token: &Arc<dyn crate::sync::CompletionToken>) {
        if !Self::scoped_sync_enabled() {
            return;
        }
        let mut producers = self.producers.lock();
        let tokens = producers.entry(base).or_default();
        tokens.retain(|t| !t.retired());
        tokens.push(Arc::clone(token));
    }

    /// Park an ownerless submission's token: waited by every scoped wait.
    pub(crate) fn record_unattributed(&self, token: Arc<dyn crate::sync::CompletionToken>) {
        let mut list = self.unattributed.lock();
        list.retain(|t| !t.retired());
        list.push(token);
    }

    /// Wait for every submission that may still touch the storage at `base`
    /// — the scoped replacement for `synchronize_all` on host-visible buffer
    /// operations. Unknown storages (and `SVOD_AMD_SCOPED_SYNC=0`) fall back
    /// to the full drain.
    pub(crate) fn wait_storage(&self, base: u64) -> Result<()> {
        if !Self::scoped_sync_enabled() {
            return self.synchronize_all();
        }
        if let Some(err) = self.poison_error() {
            return Err(err);
        }
        let tokens = {
            let mut producers = self.producers.lock();
            match producers.get_mut(&base) {
                Some(tokens) => {
                    tokens.retain(|t| !t.retired());
                    tokens.clone()
                }
                None => return self.synchronize_all(),
            }
        };
        let unattributed: Vec<_> = {
            let mut list = self.unattributed.lock();
            list.retain(|t| !t.retired());
            list.clone()
        };
        for token in tokens.iter().chain(unattributed.iter()) {
            token.wait(30_000).inspect_err(|e| self.poison(&e.to_string()))?;
        }
        Ok(())
    }

    /// Borrow the backend seam — all KFD ioctls (alloc/free/ring/wait) route
    /// through it. The allocator, queue, and connector helpers call this.
    #[inline]
    pub(crate) fn iface(&self) -> &Arc<dyn crate::amd::iface::AmdIface> {
        &self.iface
    }

    /// The KFD queue-completion event mailbox, when the backend has one. A
    /// timeline store paired with an interrupting write here wakes a blocked
    /// `WAIT_EVENTS` immediately instead of at the next poll tier.
    #[inline]
    pub(crate) fn queue_event_mailbox(&self) -> Option<crate::amd::iface::QueueEventMailbox> {
        self.iface.queue_event_mailbox()
    }

    #[inline]
    pub(crate) fn publication_checkpoint(&self, stage: crate::amd::iface::PublicationStage) -> Result<()> {
        #[cfg(test)]
        {
            self.iface.publication_checkpoint(stage)
        }
        #[cfg(not(test))]
        {
            let _ = stage;
            Ok(())
        }
    }

    /// Borrow the process-global signal pool (lazy-installed by the device
    /// factory). `None` before the factory has run; once initialized, every
    /// connector built against this core shares it.
    pub fn signal_pool(&self) -> Option<&Arc<crate::amd::signal::SignalPool>> {
        self.signal_pool.get()
    }

    /// The bounded queue-pool size (`SVOD_AMD_HW_QUEUES`; default 4 on
    /// multi-XCC hardware, 1 otherwise).
    #[inline]
    pub fn hw_queues(&self) -> usize {
        self.hw_queues
    }

    /// Borrow the process-global SDMA copy queue. `None` until the factory
    /// installs it (and thus `has_sdma_queue` is false). The allocator's
    /// device-only copy arms require it.
    pub fn copy_queue(&self) -> Option<&Arc<crate::amd::queue::AmdCopyQueue>> {
        self.copy_queue.get()
    }

    /// Install the SDMA copy queue (factory, once). Idempotent: a second call
    /// is dropped. Caller flips `has_sdma_queue` after this succeeds.
    pub fn install_copy_queue(&self, queue: Arc<crate::amd::queue::AmdCopyQueue>) {
        let _ = self.copy_queue.set(queue);
    }

    pub(crate) fn lease_queue(
        self: &Arc<Self>,
        allocator: &crate::amd::AmdAllocator,
    ) -> Result<crate::amd::connector::QueueLease> {
        self.queue_pool.acquire(self, allocator)
    }

    pub(crate) fn queue_pool(&self) -> &crate::amd::connector::QueuePool {
        &self.queue_pool
    }

    /// Install the signal pool. Called once per physical device by the
    /// runtime factory; subsequent calls are a no-op.
    pub fn install_signal_pool(&self, pool: Arc<crate::amd::signal::SignalPool>) {
        let _ = self.signal_pool.set(pool);
    }
}

impl AmdDeviceCore {
    /// Record whether an SDMA copy queue was successfully created. Called once
    /// from the device factory. When `false`, `AmdAllocator::_alloc` forces
    /// `cpu_access` so every buffer is host-visible and copies use `memmove`.
    pub fn set_has_sdma_queue(&self, present: bool) {
        self.has_sdma_queue.store(present, Ordering::Release);
    }

    /// Whether an SDMA copy queue is available.
    #[inline]
    pub fn has_sdma_queue(&self) -> bool {
        self.has_sdma_queue.load(Ordering::Acquire)
    }

    /// Enable/disable PM4 single-XCC graph capture for this device. Default OFF;
    /// set from the device factory. Tests use this to force-enable capture
    /// without mutating the process environment.
    pub fn set_pm4_graph(&self, on: bool) {
        self.pm4_graph.store(on, Ordering::Release);
    }

    /// Whether PM4 graph capture is enabled (see [`set_pm4_graph`](Self::set_pm4_graph)).
    #[inline]
    pub fn pm4_graph(&self) -> bool {
        self.pm4_graph.load(Ordering::Acquire)
    }

    /// Block in the kernel for up to `timeout_ms` waiting on **any** of the
    /// device's three events (queue completion, memory fault, hw exception).
    /// Signal polling escalates to this after a fixed spin/yield budget so a
    /// stalled wait doesn't burn CPU.
    ///
    /// Returns `Ok(Some(fault))` when a fault event fired (caller should bail
    /// with that error rather than continue polling the signal value).
    /// Returns `Ok(None)` for normal wake-ups (queue_event fired, timeout,
    /// or no event yet).
    pub fn wait_events(&self, timeout_ms: u32) -> Result<Option<Error>> {
        match self.iface.wait_events(timeout_ms) {
            Ok(fault) => {
                if let Some(error) = &fault {
                    self.poison_error_message(error);
                }
                Ok(fault)
            }
            Err(error) => {
                // Preserve the backend's typed error for the waiter that
                // observed it, while making every other owner fail fast.
                self.poison_error_message(&error);
                Err(error)
            }
        }
    }

    fn poison_error_message(&self, error: &Error) {
        match error {
            // Avoid doubling the Runtime display prefix in future failures.
            Error::Runtime { message } => self.poison(message),
            _ => self.poison(&error.to_string()),
        }
    }

    /// `true` once a fault/timeout has poisoned the device. Hot-path gate.
    #[inline]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Latch a fault: device becomes unusable, message recorded once.
    pub fn poison(&self, msg: &str) {
        let _ = self.error_msg.set(msg.to_string());
        self.poisoned.store(true, Ordering::Release);
        self.queue_pool.notify_poisoned();
    }

    /// Recorded fault if poisoned, else `None`.
    pub fn poison_error(&self) -> Option<Error> {
        self.is_poisoned().then(|| Error::Runtime {
            message: self.error_msg.get().cloned().unwrap_or_else(|| "AMD device poisoned".into()),
        })
    }

    /// Non-blocking check: did the memory-fault or hw-exception event fire
    /// since the last consumption? Issues a `WAIT_EVENTS` with `timeout=0`.
    /// Returns `Some(Error::*)` if a fault is pending, `None` otherwise.
    /// Used from `AmdSignal::wait` on a 30 s timeout to attach the real error
    /// to a stalled dispatch. (The spin-escalation path also breaks out early
    /// on a fault, but through a short blocking `wait_events`.)
    pub fn poll_faults_nonblocking(&self) -> Option<Error> {
        // Non-blocking poll = `wait_events` with timeout 0. Preserves the
        // pre-refactor contract: ioctl error / no fault → `None`; fault →
        // poison with the bare message + return `Some`.
        match self.iface.wait_events(0) {
            Ok(Some(error)) => {
                self.poison_error_message(&error);
                Some(error)
            }
            _ => None,
        }
    }
}

/// Ensure the process-wide `/dev/kfd` handle is open and return a shared
/// `Arc<OwnedFd>`. All devices in a process share one KFD fd so events
/// are visible across all of them.
pub(crate) fn ensure_global_kfd() -> Result<Arc<OwnedFd>> {
    let mut g = GLOBAL_KFD.lock();
    if let Some(fd) = g.as_ref() {
        return Ok(Arc::clone(fd));
    }
    let fd = Arc::new(open_owned("/dev/kfd")?);
    *g = Some(Arc::clone(&fd));
    Ok(fd)
}

/// Ensure the process-wide event page is allocated, bound, and mapped to
/// `node.gpu_id`:
/// - first device: allocate 0x8000 GTT|COHERENT|UNCACHED|PUBLIC, bind via
///   `CREATE_EVENT(event_page_offset=handle)`, map into the first GPU.
/// - subsequent devices: only `MAP_MEMORY_TO_GPU` the existing page into
///   their `gpu_id` (no re-alloc, no re-bind).
pub(crate) fn ensure_event_page(kfd_fd: &OwnedFd, drm_fd: &OwnedFd, node: &AmdNode) -> Result<EventPageState> {
    let mut g = EVENT_PAGE.lock();
    if let Some(ep) = g.as_ref() {
        // Reuse: map the existing page into this device's GPU page table.
        let mut gpu_id = node.gpu_id;
        let mut map_args = kfd::kfd_ioctl_map_memory_to_gpu_args {
            handle: ep.handle,
            device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
            n_devices: 1,
            n_success: 0,
        };
        if let Err(e) = unsafe { ioctl::kfd_map_memory_to_gpu(kfd_fd.as_raw_fd(), &mut map_args as *mut _) } {
            if map_args.n_success != 0 {
                unmap_event_page_from_gpu(kfd_fd, ep.handle, node.gpu_id);
            }
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_MAP_MEMORY_TO_GPU(event page reuse)", errno: e as i32 });
        }
        if map_args.n_success != 1 {
            if map_args.n_success != 0 {
                unmap_event_page_from_gpu(kfd_fd, ep.handle, node.gpu_id);
            }
            return Err(Error::AmdAllocFailed {
                reason: format!("event-page reuse mapped to {} of 1 GPU(s)", map_args.n_success),
            });
        }
        return Ok(*ep);
    }

    let allocation = alloc_event_page(kfd_fd, drm_fd, node)?;
    // Bind the page to this KFD process — only on first init.
    let mut bind =
        kfd::kfd_ioctl_create_event_args { event_page_offset: allocation.state.handle, ..Default::default() };
    if let Err(e) = unsafe { ioctl::kfd_create_event(kfd_fd.as_raw_fd(), &mut bind as *mut _) } {
        return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_EVENT(bind page)", errno: e as i32 });
    }
    let ep = allocation.commit();
    *g = Some(ep);
    Ok(ep)
}

fn unmap_event_page_from_gpu(kfd_fd: &OwnedFd, handle: u64, gpu_id: u32) {
    let mut gpu_id = gpu_id;
    let mut args = kfd::kfd_ioctl_unmap_memory_from_gpu_args {
        handle,
        device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
        n_devices: 1,
        n_success: 0,
    };
    let _ = unsafe { ioctl::kfd_unmap_memory_from_gpu(kfd_fd.as_raw_fd(), &mut args as *mut _) };
}

/// Allocate the 0x8000-byte event page (GTT-pinned, uncached, host-visible).
/// The returned guard unwinds the VA, GPU map, and KFD handle unless page
/// binding commits it to process-global state.
fn alloc_event_page<'a>(kfd_fd: &'a OwnedFd, drm_fd: &OwnedFd, node: &AmdNode) -> Result<EventPageAllocation<'a>> {
    use libc::{
        MAP_ANONYMOUS, MAP_FIXED, MAP_NORESERVE, MAP_PRIVATE, MAP_SHARED, PROT_NONE, PROT_READ, PROT_WRITE, mmap,
        munmap,
    };
    let size: usize = 0x8000;
    // SAFETY: standard libc::mmap; PROT_NONE reservation.
    let va = unsafe { mmap(std::ptr::null_mut(), size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0) };
    if va == libc::MAP_FAILED {
        return Err(Error::AmdAllocFailed { reason: "event-page VA reservation failed".into() });
    }
    let mut args = kfd::kfd_ioctl_alloc_memory_of_gpu_args {
        va_addr: va as u64,
        size: size as u64,
        gpu_id: node.gpu_id,
        flags: kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
        ..Default::default()
    };
    // SAFETY: kfd_fd is alive; args type-correct.
    if let Err(e) = unsafe { ioctl::kfd_alloc_memory_of_gpu(kfd_fd.as_raw_fd(), &mut args as *mut _) } {
        unsafe { munmap(va, size) };
        return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(event page)", errno: e as i32 });
    }
    let handle = args.handle;
    let mmap_offset = args.mmap_offset;
    let mut allocation = EventPageAllocation {
        kfd_fd,
        state: EventPageState { handle, va: va as u64, size },
        gpu_id: node.gpu_id,
        gpu_mapped: false,
        committed: false,
    };

    // Map host-visible via drm_fd at the reserved VA.
    let host = unsafe {
        mmap(va, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, drm_fd.as_raw_fd(), mmap_offset as i64)
    };
    if host == libc::MAP_FAILED || !std::ptr::eq(host, va) {
        return Err(Error::AmdAllocFailed { reason: "event-page host mmap failed".into() });
    }

    // Map into the first GPU's page table.
    let mut gpu_id = node.gpu_id;
    let mut map_args = kfd::kfd_ioctl_map_memory_to_gpu_args {
        handle,
        device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
        n_devices: 1,
        n_success: 0,
    };
    let map_result = unsafe { ioctl::kfd_map_memory_to_gpu(kfd_fd.as_raw_fd(), &mut map_args as *mut _) };
    allocation.gpu_mapped = map_args.n_success != 0;
    if let Err(e) = map_result {
        return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_MAP_MEMORY_TO_GPU(event page)", errno: e as i32 });
    }
    if map_args.n_success != 1 {
        return Err(Error::AmdAllocFailed {
            reason: format!("event page mapped to {} of 1 GPU(s)", map_args.n_success),
        });
    }

    Ok(allocation)
}

/// Allocate a scratch buffer sized for `private_segment_size` bytes per
/// thread and compute the packed `COMPUTE_TMPRING_SIZE` value. Returns
/// `(scratch_gpu_va, scratch_size, tmpring_size, rounded_size_per_thread,
/// handle, aql_desc)`.
///
/// Sizing (gfx11/12):
/// - `lanes_per_wave = 64` (scratch lane stride is wave64-aligned per AMD)
/// - `mem_alignment_size = 256`
/// - `size_per_thread = round_up(private_segment_size, 4)` (= 256/64)
/// - `cu_cnt = simd_count / simd_per_cu / xccs`
/// - `size_per_xcc = size_per_thread * lanes_per_wave * max_slots_scratch_cu * cu_cnt`
/// - `total = size_per_xcc * xccs` (page-aligned for KFD)
///
/// `COMPUTE_TMPRING_SIZE` packs `WAVES` (bits 0-11) and `WAVESIZE`
/// (bits 12-26 on gfx11):
/// - `wave_scratch = ceildiv(lanes_per_wave * size_per_thread, 256)`
/// - `num_waves = (size_per_xcc / (wave_scratch * 256)) / se_cnt`
/// - `max_scratch_waves = cu_cnt * max_slots_scratch_cu * xccs`
/// - `WAVES = min(num_waves, max_scratch_waves)`, `WAVESIZE = wave_scratch`
pub(crate) fn alloc_scratch(
    iface: &Arc<dyn crate::amd::iface::AmdIface>,
    node: &AmdNode,
    arch: &AmdArch,
    private_segment_size: u32,
) -> Result<(u64, usize, u32, u32, u64, AqlScratchDesc)> {
    const LANES_PER_WAVE: u32 = 64;
    const PAGE: usize = 0x1000;
    // gfx9 (CDNA) scratch is 1024-byte aligned; gfx11/12 (RDNA) use 256.
    let mem_alignment_size: u32 = if arch.is_cdna() { 1024 } else { 256 };

    let xccs = node.num_xcc.max(1);
    let simd_per_cu = node.simd_per_cu.max(1);
    let cu_cnt = ((node.simd_count.max(1) / simd_per_cu) / xccs).max(1);
    let max_slots = node.max_slots_scratch_cu.max(1);
    let se_cnt = (node.array_count.max(1) / node.simd_arrays_per_engine.max(1) / xccs).max(1);
    // ROCr sizes scratch for `AlignUp(cu_count, shader_engines) * MaxSlotsScratchCU`
    // slots PER XCC (amd_aql_queue.cpp `calc_device_slots`), NOT raw `cu_count`.
    // The CP rounds its per-dispatch occupancy demand up to the shader-engine
    // boundary; sizing from raw `cu_cnt` under-provisions on a harvested 8-XCC
    // part, so a high-occupancy dispatch traps with an insufficient-scratch
    // exception (0x401) and the CP HALTS the queue waiting on `queue_inactive_
    // signal` — which we never service, so it hangs forever with no fault.
    let cu_slots = cu_cnt.next_multiple_of(se_cnt);

    // Round up to the per-lane alignment stride.
    let size_per_thread = private_segment_size.max(1).next_multiple_of(mem_alignment_size / LANES_PER_WAVE);
    let size_per_xcc =
        (size_per_thread as usize) * (LANES_PER_WAVE as usize) * (max_slots as usize) * (cu_slots as usize);
    let total = (size_per_xcc * xccs as usize).next_multiple_of(PAGE);

    // KFD alloc as plain VRAM (GPU-only; no host access needed — the GPU writes
    // register spills here and reads them back). Plain VRAM = no EXECUTABLE, no
    // PUBLIC (`cpu_access=false` keeps PUBLIC off); see `AllocKind::DeviceVram`.
    let r = iface.alloc_raw(
        total,
        crate::amd::iface::AllocKind::DeviceVram { executable: false },
        crate::amd::va_registry::AllocTag::Scratch,
        /*cpu_access=*/ false,
        /*zero=*/ false,
    )?;
    let va = r.gpu_va;
    let total = r.size;
    let handle = r.handle;

    // gfx9 divides scratch evenly across SEs (1); gfx11/12 divide by se_cnt.
    let wave_scratch = (LANES_PER_WAVE * size_per_thread).div_ceil(mem_alignment_size);
    let max_scratch_waves = cu_slots * max_slots * xccs;
    let se_div = if arch.is_cdna() { 1 } else { se_cnt };
    let num_waves = ((size_per_xcc as u32) / (wave_scratch * mem_alignment_size)) / se_div;
    // COMPUTE_TMPRING_SIZE.WAVES must be a multiple of the per-XCC shader-engine
    // count (amd_aql_queue.cpp asserts `WAVES % (banks/xcc) == 0`); round down to
    // that boundary (≥ one engine's worth) so the CP accepts the wave count.
    let waves = (num_waves.min(max_scratch_waves) / se_cnt).max(1) * se_cnt;
    let tmpring_size = pack_tmpring(waves, wave_scratch, arch);

    // Every AQL queue reads the descriptor out of `amd_queue_t` — multi-XCC
    // CDNA always, gfx11+ under `SVOD_AMD_AQL=1`. PM4 queues program scratch
    // via per-dispatch `SET_SH_REG` instead and ignore it, but synthesizing
    // it unconditionally is harmless and closes the forced-AQL gap where
    // gfx11 kernels needing scratch read an all-zero SRD. Explicit per
    // generation (ROCr `InitScratchSRD` switches the same way): a future arch
    // must pick its own V# layout here, not inherit one from an else-bucket.
    let aql_desc = match arch.gfx_major() {
        9 => AqlScratchDesc::gfx9(va, size_per_xcc, tmpring_size, private_segment_size),
        // gfx12's V# differs from gfx11 only in compression bits that stay
        // zero; the TMPRING packing difference lives in `pack_tmpring`.
        11 | 12 => AqlScratchDesc::gfx11(va, size_per_xcc, tmpring_size, size_per_thread),
        major => unreachable!("no AQL scratch V# layout for gfx{major}"),
    };

    Ok((va, total, tmpring_size, size_per_thread, handle, aql_desc))
}

/// Pack `COMPUTE_TMPRING_SIZE`: WAVES in bits 0..12, WAVESIZE at bit 12 with an
/// arch-specific field width — gfx9 13b, gfx11 15b, gfx12 18b.
pub(crate) fn pack_tmpring(waves: u32, wave_scratch: u32, arch: &AmdArch) -> u32 {
    let wavesize_mask: u32 = if arch.is_cdna() {
        0x1FFF
    } else if arch.is_rdna4() {
        0x3FFFF
    } else {
        0x7FFF
    };
    (waves & 0xFFF) | ((wave_scratch & wavesize_mask) << 12)
}

pub(crate) fn open_owned(path: &str) -> Result<OwnedFd> {
    match open(path, OFlag::O_RDWR | OFlag::O_CLOEXEC, Mode::empty()) {
        Ok(fd) => Ok(fd),
        Err(nix::errno::Errno::ENOENT) | Err(nix::errno::Errno::EACCES) => {
            Err(Error::NoAmdGpu { reason: format!("cannot open {path}") })
        }
        Err(e) => Err(Error::AmdIoctl { ioctl: "open", errno: e as i32 }),
    }
}
