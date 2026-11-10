//! `AmdIface`: the device-backend seam.
//!
//! `AmdDeviceCore` no longer issues KFD ioctls directly — it routes every
//! memory alloc/free, queue ring setup/teardown, and event wait through an
//! `Arc<dyn AmdIface>`. The only implementor today is [`KfdIface`], a verbatim
//! relocation of the KFD-direct path that previously lived inline across
//! `allocator.rs`, `device.rs`, and `queue.rs`. KFD-specific state (the kfd /
//! drm fds, KFD ABI version, event ids, and the per-process event page) lives
//! on `KfdIface`, off the core. A future userspace AM driver becomes a second
//! `AmdIface` implementor without touching any call site above the seam.
//!
//! This is a pure refactor: flag composition, ioctl argument layout, error
//! mapping, and the fault-message strings are reproduced bit-for-bit.

#![cfg(unix)]

use std::os::fd::{AsRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use libc::{
    MAP_ANONYMOUS, MAP_FIXED, MAP_NORESERVE, MAP_PRIVATE, MAP_SHARED, PROT_NONE, PROT_READ, PROT_WRITE, mmap, munmap,
};
use tracing::debug;

use crate::amd::sys::{ioctl, kfd};
use crate::amd::topology::AmdNode;
use crate::amd::va_registry::{AllocTag, VaRegistry};
use crate::error::{Error, Result};

/// Size of the doorbell MMIO window mapped per queue (two 4 KiB pages).
const DOORBELL_PAGE_BYTES: usize = 0x2000;
/// The GPU page a large allocation should be mapped with.
pub(crate) const HUGE_PAGE: usize = 0x20_0000;

/// Backend seam for the AMD device. Every KFD ioctl that `AmdDeviceCore` (and
/// its allocator / queue / connector helpers) needs is funnelled through one
/// of these methods — five required, four defaulted — so a second backend
/// (the userspace AM driver) can be dropped in without editing any call site.
pub trait AmdIface: Send + Sync + std::fmt::Debug {
    /// Reserve a host VA, KFD-allocate `size` bytes per `kind`, optionally map
    /// host-visible, and bind it into the GPU page table. `zero` zero-fills the
    /// allocation, but this seam can only honor it for host-mapped buffers —
    /// `zero=true` with `cpu_access=false` is rejected with an error (the caller
    /// owns device-zeroing via the SDMA copy queue). `tag` records the
    /// allocation's purpose in the VA registry so a later fault can be resolved
    /// back to it.
    fn alloc_raw(
        &self,
        size: usize,
        kind: AllocKind,
        tag: AllocTag,
        cpu_access: bool,
        zero: bool,
    ) -> Result<AllocResult>;
    /// Unmap from the GPU, drop the host mapping, and free the KFD allocation.
    /// Best-effort (called from `Drop`); ioctl errors are swallowed.
    fn free_raw(&self, gpu_va: u64, size: usize, handle: u64);
    /// Create a KFD compute/SDMA queue over a pre-allocated ring + GART and
    /// return its queue id + mmapped doorbell.
    fn setup_ring(&self, desc: &RingDesc) -> Result<QueueHandle>;
    /// Destroy the in-kernel queue object and `munmap` the queue's doorbell
    /// page (`doorbell_base` is the mmap base from [`QueueHandle`]).
    fn teardown_ring(&self, queue_id: u32, doorbell_base: NonNull<u8>) -> Result<QueueTeardown>;
    /// Block up to `timeout_ms` on the device's completion + fault events.
    /// `Ok(Some(Error::GpuFault{..}))` on a memory fault,
    /// `Ok(Some(Error::Runtime{..}))` on a hardware exception, `Ok(None)` on a
    /// normal wake-up/timeout, `Err` if the WAIT_EVENTS ioctl itself failed.
    ///
    /// `age` is the queue event's age this waiter last saw (KFD ≥ 1.14): the
    /// call returns at once if the event fired since, and writes the new age
    /// back, so every waiter sees each completion however many share the
    /// auto-reset event. A wait starts at 1, the age of an event that never
    /// fired; 0 opts out (ROCr `InterruptSignal::WaitRelaxed`).
    fn wait_events(&self, timeout_ms: u32, age: &mut u64) -> Result<Option<Error>>;

    /// The KFD queue-completion event mailbox, when this backend has one.
    /// A completion packet that writes `event_id` there and raises an interrupt
    /// wakes a blocked `WAIT_EVENTS` immediately instead of leaving it to the
    /// poll tier (tinygrad `AMDComputeQueue.signal`, `ops_amd.py:391-393`).
    /// `None` on backends with no KFD event page (AM, host mocks), which then
    /// rely on the coherent GTT slot alone.
    fn queue_event_mailbox(&self) -> Option<QueueEventMailbox> {
        None
    }

    /// Fault-injection checkpoint around queue publication. Production
    /// backends keep the default no-op; the host mock scripts failures here to
    /// prove reservation rollback and post-doorbell poisoning.
    fn publication_checkpoint(&self, _stage: PublicationStage) -> Result<()> {
        Ok(())
    }

    /// `AMDKFD_IOC_UPDATE_QUEUE` with the given `queue_percentage`: `0` unmaps
    /// the queue from the hardware scheduler, `> 0` remaps it — and a remap is
    /// the only event that makes CP firmware re-read the queue's `amd_queue_t`
    /// descriptor (scratch SRD/TMPRING), which it caches at queue-connect
    /// (ROCr `AqlQueue::Suspend/Resume`). KFD re-validates the ring on every
    /// call, so the ring VA and size must be passed again. Backends without a
    /// KFD queue scheduler (host mocks) accept the default no-op.
    fn update_queue_percentage(&self, _queue_id: u32, _ring_gpu: u64, _ring_size: u32, _percentage: u32) -> Result<()> {
        Ok(())
    }

    /// Flush the host's writes to device-local memory out of HDP, the host
    /// data path, and wait until the flush has landed (CLR's
    /// `HDP_MEM_FLUSH_CNTL` write; ROCr only exposes the register to its
    /// clients as `HDP_flush_`). Called through [`Hdp`] only. A no-op
    /// where the backend has no flush register.
    fn flush_hdp(&self) {}
}

/// HDP, the host data path: host stores into VRAM through the BAR may sit in
/// it until it is flushed, so no GPU engine may read them before a flush.
///
/// Kernargs and command buffers (the kernarg arena, graph and plan kernargs,
/// PM4 IBs and AQL control programs) live where the device reads them
/// fastest, as CLR places them: VRAM on a multi-XCC part, where reads from
/// system memory slow every kernel, and uncached system memory elsewhere,
/// where draining the host's stores to VRAM costs more than it saves. With
/// them in VRAM every doorbell flushes; otherwise the submission path writes
/// only system memory and a doorbell flushes only what host writes deferred.
///
/// Other host stores into VRAM are rare and explicit (code upload, CWSR
/// headers, zero-fill, copy-in, host transfers): each flushes with
/// [`Hdp::flush`] once its stores are done. A [`crate::buffer::HostWrite`]
/// into VRAM marks its stores [`Hdp::defer`]red when it drops, and the
/// device's next doorbell flushes them once for every guard since the last.
/// A doorbell counts a deferral flushed only once its own flush has landed,
/// so one racing another's flush still flushes.
///
/// It also counts the stores that reach device memory behind the GPU caches,
/// which a lane's caches may hold stale lines for: these host stores and the
/// copy engine's ([`Self::note_dma_write`]). A lane invalidates its caches
/// before its next dispatch once the count passed the one it last invalidated
/// at (`PoolQueue::cache_invalidate`). Kernargs and command buffers are not
/// counted: uncached system memory bypasses L2, and a multi-XCC part keeps L2
/// coherent with host stores to its local VRAM (CLR's device kernargs).
#[derive(Debug)]
pub(crate) struct Hdp {
    iface: Arc<dyn AmdIface>,
    deferred: AtomicU64,
    flushed: AtomicU64,
    behind_caches: AtomicU64,
    submissions_in_vram: bool,
}

impl Hdp {
    pub(crate) fn new(iface: Arc<dyn AmdIface>, submissions_in_vram: bool) -> Self {
        Self {
            iface,
            deferred: AtomicU64::new(0),
            flushed: AtomicU64::new(0),
            // Above any lane's start, so a new lane's first dispatch invalidates.
            behind_caches: AtomicU64::new(1),
            submissions_in_vram,
        }
    }

    /// Stores behind the GPU caches so far.
    pub(crate) fn writes_behind_caches(&self) -> u64 {
        self.behind_caches.load(Ordering::Acquire)
    }

    /// The copy engine is about to store behind the GPU caches.
    pub(crate) fn note_dma_write(&self) {
        self.behind_caches.fetch_add(1, Ordering::AcqRel);
    }

    /// Whether kernargs and command buffers live in VRAM.
    pub(crate) fn submissions_in_vram(&self) -> bool {
        self.submissions_in_vram
    }

    /// Flush this thread's VRAM stores so far: drain its write-combining
    /// buffers, then flush HDP and read the flush back.
    pub(crate) fn flush(&self) {
        std::sync::atomic::fence(Ordering::SeqCst);
        self.iface.flush_hdp();
        self.behind_caches.fetch_add(1, Ordering::AcqRel);
    }

    /// This thread's VRAM stores so far are done; the device's next doorbell
    /// flushes them. The fence drains its write-combining buffers first.
    pub(crate) fn defer(&self) {
        std::sync::atomic::fence(Ordering::SeqCst);
        self.deferred.fetch_add(1, Ordering::Release);
        self.behind_caches.fetch_add(1, Ordering::AcqRel);
    }

    /// Run by every doorbell after its SeqCst fence: flush the submission's
    /// own stores when they went to VRAM, else the stores host writes
    /// deferred since the last doorbell.
    pub(crate) fn flush_for_doorbell(&self) {
        let deferred = self.deferred.load(Ordering::Acquire);
        if self.submissions_in_vram || self.flushed.load(Ordering::Acquire) < deferred {
            self.iface.flush_hdp();
            self.flushed.fetch_max(deferred, Ordering::Release);
        }
    }
}

/// Address of a KFD event's mailbox slot plus the id written into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueEventMailbox {
    pub address: u64,
    pub event_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicationStage {
    AfterReservation,
    BeforeDoorbell,
    AfterDoorbell,
}

/// Result after KFD has definitively stopped a queue. A leaked doorbell mapping
/// is a host-resource leak, but no longer requires GPU backing quarantine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueTeardown {
    Complete,
    DoorbellLeaked { errno: i32 },
}

/// Allocation flavor — selects the KFD flag set built in [`AmdIface::alloc_raw`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocKind {
    /// Device VRAM. `executable` adds the EXECUTABLE bit (set for general
    /// buffers, cleared for scratch). PUBLIC is derived from `cpu_access`.
    DeviceVram { executable: bool },
    /// GTT-pinned, host-visible, uncached system memory: everything the host
    /// writes per submission (rings, GART, signal slots, kernargs, command
    /// buffers). The GPU reads it snooped, through no HDP, and `UNCACHED`
    /// maps it MTYPE UC on every arch, so no GPU L2 keeps a stale copy of a
    /// reused slot. `COHERENT` alone maps NC, cached in L2, on gfx12
    /// (`gmc_v12_0_get_vm_pte`) and on a gfx942 APU, where each XCC's L2
    /// would cache it apart.
    UncachedGtt,
}

/// Result of [`AmdIface::alloc_raw`]: everything `RawBuffer::AmdDevice` needs
/// except the owning device handle (the allocator adds that at the call site).
#[derive(Debug)]
pub struct AllocResult {
    pub gpu_va: u64,
    pub host_ptr: Option<NonNull<u8>>,
    pub handle: u64,
    pub size: usize,
}

// SAFETY: `host_ptr` is a raw mapping pointer with no thread affinity; access
// is synchronized by the scheduler / single-owner invariant exactly as
// `RawBuffer` asserts. Needed only so `Result<AllocResult>` crosses the trait.
unsafe impl Send for AllocResult {}

/// Inputs for [`AmdIface::setup_ring`]: the caller pre-allocates the ring /
/// GART / EOP / ctx-save backings (above the seam) and hands their GPU VAs +
/// sizes plus the descriptor offsets here.
pub struct RingDesc {
    pub ring_gpu: u64,
    pub gart_gpu: u64,
    pub wptr_offset: u64,
    pub rptr_offset: u64,
    pub eop_gpu: u64,
    pub eop_size: u64,
    pub ctx_gpu: u64,
    pub ctx_save_restore_size: u32,
    pub ctl_stack_size: u32,
    pub ring_size: usize,
    pub gpu_id: u32,
    pub queue_type: u32,
}

/// Result of [`AmdIface::setup_ring`]: the kernel-assigned queue id and the
/// per-queue doorbell pointer.
pub struct QueueHandle {
    pub queue_id: u32,
    /// mmap base of the doorbell page, kept so the page can be `munmap`'d on
    /// teardown (each queue maps its own page).
    pub doorbell_base: NonNull<u8>,
    pub doorbell: NonNull<u64>,
}

// SAFETY: `doorbell` is an MMIO mapping pointer; the owning `QueueInner`
// already asserts `Send`/`Sync` over it. Needed so `Result<QueueHandle>`
// crosses the trait.
unsafe impl Send for QueueHandle {}

/// KFD-direct backend. Owns the per-process/per-device KFD state that used to
/// live on `AmdDeviceCore`.
#[derive(Debug)]
pub struct KfdIface {
    /// Shared `/dev/kfd` handle (one per process).
    kfd_fd: Arc<OwnedFd>,
    drm_fd: OwnedFd,
    /// Its own clone of the topology node (gpu_id, drm minor, etc.).
    node: AmdNode,
    kfd_version: (u32, u32),
    /// VA + size of the GTT-pinned event page (held to keep it mapped).
    #[allow(dead_code)]
    event_page_va: u64,
    #[allow(dead_code)]
    event_page_size: usize,
    /// KFD ids for the SIGNAL events used by queue completion / fault paths.
    queue_event_id: u32,
    #[allow(dead_code)]
    queue_event_slot_index: u32,
    queue_event_mailbox_ptr: u64,
    mem_fault_event_id: u32,
    hw_fault_event_id: u32,
    /// VA → allocation registry: every `alloc_raw` range, plus a bounded ring
    /// of recently-freed ranges, so a fault VA can be resolved to its owning
    /// (or stale, or nearest) allocation. See [`crate::amd::va_registry`].
    va: VaRegistry,
    /// How the host flushes HDP: KFD's remap of the flush register, else a
    /// VRAM readback.
    hdp: Option<HdpFlush>,
    /// One-shot latch so the terminal fault is logged at `error` exactly once,
    /// even though `wait_events(0)` may re-observe the (non-auto-reset) fault
    /// event on subsequent poll-fault calls.
    fault_logged: AtomicBool,
}

impl KfdIface {
    /// Open `/dev/kfd` + the node's DRM render fd, acquire the VM, enable the
    /// runtime, bind the event page, and create the three completion / fault
    /// events. Verbatim move of the bring-up block from `AmdDevice::open_uncached`.
    pub fn open(node: &AmdNode) -> Result<Self> {
        let kfd_fd = crate::amd::device::ensure_global_kfd()?;
        let drm_path = format!("/dev/dri/renderD{}", node.drm_render_minor);
        let drm_fd = crate::amd::device::open_owned(drm_path.as_str())?;

        // GET_VERSION captures the KFD ABI version so we can gate RUNTIME_ENABLE
        // (which only exists on kfd >= 1.14).
        let mut ver_args = kfd::kfd_ioctl_get_version_args { major_version: 0, minor_version: 0 };
        if let Err(e) = unsafe { ioctl::kfd_get_version(kfd_fd.as_raw_fd(), &mut ver_args as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_GET_VERSION", errno: e as i32 });
        }
        let kfd_version = (ver_args.major_version, ver_args.minor_version);

        // Opening the fd queues the root page directory clear on SDMA, and
        // ACQUIRE_VM's wait for it skips that BOOKKEEP fence (`amdgpu_sync_resv`
        // stops at READ). The clear then lands over the first PDEs KFD writes
        // with the CPU, and the walker retries forever until the GPU resets.
        // `amdgpu_flush` runs on any close and drains the VM's job queues;
        // nothing exposes the jobs' completion, so a pause covers their execution.
        let flushed = drm_fd
            .try_clone()
            .map_err(|e| Error::AmdIoctl { ioctl: "dup(render node)", errno: e.raw_os_error().unwrap_or(0) })?;
        drop(flushed);
        std::thread::sleep(std::time::Duration::from_millis(1));

        // ACQUIRE_VM tells KFD to register this DRM fd as the process's VM
        // for this GPU. Required before any alloc/map ioctls.
        let mut args = kfd::kfd_ioctl_acquire_vm_args { drm_fd: drm_fd.as_raw_fd() as u32, gpu_id: node.gpu_id };
        // SAFETY: kfd_fd is a valid open fd; `args` is a well-typed ioctl
        // argument matching the AMDKFD_IOC_ACQUIRE_VM signature.
        let rc = unsafe { ioctl::kfd_acquire_vm(kfd_fd.as_raw_fd(), &mut args as *mut _) };
        if let Err(e) = rc {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_ACQUIRE_VM", errno: e as i32 });
        }

        // RUNTIME_ENABLE — only on KFD >= 1.14; older kernels reject the
        // ioctl with ENOTTY. A mask without the enable bit disables instead.
        // KFD enables a process once (EBUSY after) and only before its first
        // queue (EEXIST after), so the first device to open enables it and
        // a later one finding it enabled goes on.
        if kfd_version >= (1, 14) {
            static ENABLED: OnceLock<std::result::Result<(), nix::errno::Errno>> = OnceLock::new();
            let enabled = ENABLED.get_or_init(|| {
                let mut rt = kfd::kfd_ioctl_runtime_enable_args {
                    mode_mask: kfd::KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK,
                    ..Default::default()
                };
                match unsafe { ioctl::kfd_runtime_enable(kfd_fd.as_raw_fd(), &mut rt as *mut _) } {
                    Ok(_) | Err(nix::errno::Errno::EBUSY) => Ok(()),
                    Err(e) => Err(e),
                }
            });
            if let Err(e) = enabled {
                return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_RUNTIME_ENABLE", errno: *e as i32 });
            }
        }

        // Event-page setup: allocated and bound
        // exactly once per process; subsequent devices reuse it by calling
        // `MAP_MEMORY_TO_GPU` for their `gpu_id`. Without the bound event page,
        // AMDKFD_IOC_CREATE_QUEUE returns EINVAL.
        let (event_page_va, event_page_size) = {
            let ep = crate::amd::device::ensure_event_page(&kfd_fd, &drm_fd, node)?;
            (ep.va, ep.size)
        };
        let mut qe = kfd::kfd_ioctl_create_event_args {
            event_type: kfd::KFD_IOC_EVENT_SIGNAL,
            auto_reset: 1,
            ..Default::default()
        };
        if let Err(e) = unsafe { ioctl::kfd_create_event(kfd_fd.as_raw_fd(), &mut qe as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_EVENT(queue signal)", errno: e as i32 });
        }
        let mut queue_event = EventGuard::new(kfd_fd.as_raw_fd(), qe.event_id);
        let mut mem_event =
            kfd::kfd_ioctl_create_event_args { event_type: kfd::KFD_IOC_EVENT_MEMORY, ..Default::default() };
        if let Err(e) = unsafe { ioctl::kfd_create_event(kfd_fd.as_raw_fd(), &mut mem_event as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_EVENT(mem fault)", errno: e as i32 });
        }
        let mut memory_event = EventGuard::new(kfd_fd.as_raw_fd(), mem_event.event_id);
        let mut hw_event =
            kfd::kfd_ioctl_create_event_args { event_type: kfd::KFD_IOC_EVENT_HW_EXCEPTION, ..Default::default() };
        if let Err(e) = unsafe { ioctl::kfd_create_event(kfd_fd.as_raw_fd(), &mut hw_event as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_EVENT(hw fault)", errno: e as i32 });
        }
        let mut hardware_event = EventGuard::new(kfd_fd.as_raw_fd(), hw_event.event_id);
        queue_event.disarm();
        memory_event.disarm();
        hardware_event.disarm();

        // The mailbox sits at event_page + slot_index * 8. SDMA fence packets
        // write the queue event_id here to wake up `WAIT_EVENTS` from `sleep()`.
        let queue_event_mailbox_ptr = event_page_va + (qe.event_slot_index as u64) * 8;

        debug!(
            node = node.node_id,
            gpu_id = node.gpu_id,
            kfd_version = ?kfd_version,
            queue_event_id = qe.event_id,
            mem_fault_event_id = mem_event.event_id,
            hw_fault_event_id = hw_event.event_id,
            "KfdIface opened"
        );
        // ROCr's libhsakmt warns identically (`topology.c`): gfx1151 on a KFD
        // ABI below 1.20 "may result in faults, crashes and other application
        // instability". Surface it once here — MES wedges on this part are
        // expensive to debug without this hint.
        if node.gfx_target_version == 110501 && kfd_version < (1, 20) {
            tracing::warn!(
                kfd_version = ?kfd_version,
                "gfx1151 recommends KFD ABI 1.20+; older ABIs are known to fault and destabilize applications"
            );
        }

        let mmio_remap = MmioRemapPage::map(&kfd_fd, node.gpu_id);
        let mut iface = Self {
            kfd_fd,
            drm_fd,
            node: node.clone(),
            kfd_version,
            event_page_va,
            event_page_size,
            queue_event_id: qe.event_id,
            queue_event_slot_index: qe.event_slot_index,
            queue_event_mailbox_ptr,
            mem_fault_event_id: mem_event.event_id,
            hw_fault_event_id: hw_event.event_id,
            va: VaRegistry::default(),
            hdp: mmio_remap.map(HdpFlush::Register),
            fault_logged: AtomicBool::new(false),
        };
        if iface.hdp.is_none() {
            // No remap page (an SR-IOV VF, whose kernel keeps the flush
            // register to itself; larger host pages): read a VRAM word back,
            // as ROCr (`PcieWcFlush`) and CLR (`DeviceKernelArgsReadback`) do.
            let word = iface.alloc_raw(
                0x1000,
                AllocKind::DeviceVram { executable: false },
                AllocTag::HdpReadback,
                true,
                true,
            )?;
            iface.hdp = Some(HdpFlush::Readback(word));
        }
        Ok(iface)
    }
}

impl Drop for KfdIface {
    fn drop(&mut self) {
        match self.hdp.take() {
            Some(HdpFlush::Register(page)) => {
                // SAFETY: the page was mapped by `MmioRemapPage::map`; queues that
                // flushed through it hold this iface, so none outlives it.
                unsafe { munmap(page.base.as_ptr().cast(), MMIO_REMAP_PAGE_BYTES) };
                self.free_kfd(page.handle);
            }
            Some(HdpFlush::Readback(word)) => self.free_raw(word.gpu_va, word.size, word.handle),
            None => {}
        }
        for event_id in [self.queue_event_id, self.mem_fault_event_id, self.hw_fault_event_id] {
            let mut args = kfd::kfd_ioctl_destroy_event_args { event_id, pad: 0 };
            let _ = unsafe { ioctl::kfd_destroy_event(self.kfd_fd.as_raw_fd(), &mut args as *mut _) };
        }
    }
}

/// How the host pushes its VRAM stores out of HDP before a doorbell.
#[derive(Debug)]
enum HdpFlush {
    /// Write the flush register through KFD's remap page.
    Register(MmioRemapPage),
    /// Store to a host-visible VRAM word and read it back. A BAR read cannot
    /// pass the posted writes before it, and AMD documents one readback as
    /// flushing every earlier store (`HSA_AMD_AGENT_INFO_PCIE_STORES_NEED_FLUSH`;
    /// CLR's `FlushSignalStores` flushes many with one).
    Readback(AllocResult),
}

// SAFETY: the readback word is a mapping with no thread affinity; concurrent
// flushes store the same value and only read it back.
unsafe impl Sync for HdpFlush {}

impl HdpFlush {
    fn flush(&self) {
        match self {
            HdpFlush::Register(page) => page.flush(),
            HdpFlush::Readback(word) => {
                let word = word.host_ptr.expect("host-visible readback word").cast::<u32>().as_ptr();
                // SAFETY: the word is this iface's own mapped VRAM allocation.
                unsafe {
                    std::ptr::write_volatile(word, 0);
                    std::sync::atomic::fence(Ordering::SeqCst);
                    std::ptr::read_volatile(word);
                }
            }
        }
    }
}

/// Size of KFD's MMIO-remap page; KFD offers it only on 4 KiB host pages.
const MMIO_REMAP_PAGE_BYTES: usize = 0x1000;

/// KFD's user mapping of the device's MMIO-remap page, which exposes the HDP
/// flush registers (libhsakmt `map_mmio`, read by ROCr as `HDP_flush_`).
///
/// A write to `HDP_MEM_FLUSH_CNTL` flushes HDP from the host, without the CP's
/// `GPU_HDP_FLUSH_REQ/DONE` handshake: that register pair is one channel per
/// bit shared by every queue of every process, and its DONE bit toggles per
/// request, so concurrent requesters on one bit can wait forever (four
/// lanes on gfx942 stalled 14 of 14 runs and ended in a GPU reset).
#[derive(Debug)]
struct MmioRemapPage {
    base: NonNull<u32>,
    handle: u64,
}

// SAFETY: an MMIO mapping with no thread affinity; a flush is one register
// write any thread may issue.
unsafe impl Send for MmioRemapPage {}
unsafe impl Sync for MmioRemapPage {}

impl MmioRemapPage {
    /// Allocate and map the remap page, or `None` where KFD has none to give
    /// (larger host pages, an SR-IOV VF, whose kernel keeps the register, an
    /// older KFD): the host then flushes by readback.
    fn map(kfd_fd: &OwnedFd, gpu_id: u32) -> Option<Self> {
        // SAFETY: sysconf has no preconditions.
        if unsafe { libc::sysconf(libc::_SC_PAGESIZE) } != MMIO_REMAP_PAGE_BYTES as libc::c_long {
            return None;
        }
        let va = reserve_va(MMIO_REMAP_PAGE_BYTES).ok()?;
        let mut args = kfd::kfd_ioctl_alloc_memory_of_gpu_args {
            va_addr: va as u64,
            size: MMIO_REMAP_PAGE_BYTES as u64,
            gpu_id,
            flags: kfd::KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT,
            ..Default::default()
        };
        // SAFETY: kfd_fd is open; args is well-typed for ALLOC_MEMORY_OF_GPU.
        if let Err(errno) = unsafe { ioctl::kfd_alloc_memory_of_gpu(kfd_fd.as_raw_fd(), &mut args as *mut _) } {
            unsafe { munmap(va, MMIO_REMAP_PAGE_BYTES) };
            debug!(?errno, "no KFD MMIO remap page; the host flushes HDP by readback");
            return None;
        }
        // SAFETY: KFD hands out the remap page through /dev/kfd at `mmap_offset`.
        let p = unsafe {
            mmap(
                va,
                MMIO_REMAP_PAGE_BYTES,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_FIXED,
                kfd_fd.as_raw_fd(),
                args.mmap_offset as i64,
            )
        };
        if p == libc::MAP_FAILED {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            let mut free = kfd::kfd_ioctl_free_memory_of_gpu_args { handle: args.handle };
            // SAFETY: the handle came from the allocation above.
            let _ = unsafe { ioctl::kfd_free_memory_of_gpu(kfd_fd.as_raw_fd(), &mut free as *mut _) };
            unsafe { munmap(va, MMIO_REMAP_PAGE_BYTES) };
            tracing::warn!(errno, "mmap(KFD MMIO remap) failed; the host flushes HDP by readback");
            return None;
        }
        Some(Self { base: NonNull::new(p.cast())?, handle: args.handle })
    }

    /// Flush HDP: write `HDP_MEM_FLUSH_CNTL`, then read it back so the flush
    /// has landed before the caller's doorbell write (CLR `rocvirtual.cpp`).
    fn flush(&self) {
        let offset = kfd::kfd_mmio_remap_KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL as usize / 4;
        // SAFETY: the register lies inside the mapped page.
        unsafe {
            let reg = self.base.as_ptr().add(offset);
            std::ptr::write_volatile(reg, 1);
            std::ptr::read_volatile(reg);
        }
    }
}

struct EventGuard {
    kfd_fd: i32,
    event_id: Option<u32>,
}

impl EventGuard {
    fn new(kfd_fd: i32, event_id: u32) -> Self {
        Self { kfd_fd, event_id: Some(event_id) }
    }

    fn disarm(&mut self) {
        self.event_id = None;
    }
}

impl Drop for EventGuard {
    fn drop(&mut self) {
        let Some(event_id) = self.event_id else { return };
        let mut args = kfd::kfd_ioctl_destroy_event_args { event_id, pad: 0 };
        let _ = unsafe { ioctl::kfd_destroy_event(self.kfd_fd, &mut args as *mut _) };
    }
}

impl AmdIface for KfdIface {
    fn flush_hdp(&self) {
        if let Some(hdp) = &self.hdp {
            hdp.flush();
        }
    }

    fn queue_event_mailbox(&self) -> Option<QueueEventMailbox> {
        Some(QueueEventMailbox { address: self.queue_event_mailbox_ptr, event_id: self.queue_event_id })
    }

    fn alloc_raw(
        &self,
        size: usize,
        kind: AllocKind,
        tag: AllocTag,
        cpu_accessible: bool,
        zero_init: bool,
    ) -> Result<AllocResult> {
        // KFD VA reservation + map are page-granular; a 0-byte mmap is EINVAL.
        let size = size.max(1).next_multiple_of(0x1000);
        let flags = compose_flags(kind, cpu_accessible);
        let va = reserve_va(size)?;
        let mut args = kfd::kfd_ioctl_alloc_memory_of_gpu_args {
            va_addr: va as u64,
            size: size as u64,
            gpu_id: self.node.gpu_id,
            flags,
            ..Default::default()
        };
        if let Err(e) = unsafe { ioctl::kfd_alloc_memory_of_gpu(self.kfd_fd.as_raw_fd(), &mut args as *mut _) } {
            unsafe { munmap(va as *mut _, size) };
            return Err(map_alloc_err(e, cpu_accessible));
        }
        let mem_handle = args.handle;
        let mmap_offset = args.mmap_offset;

        let host_ptr = if cpu_accessible {
            let p = unsafe {
                mmap(
                    va as *mut _,
                    size,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED,
                    self.drm_fd.as_raw_fd(),
                    mmap_offset as i64,
                )
            };
            if p == libc::MAP_FAILED || !std::ptr::eq(p, va) {
                self.free_kfd(mem_handle);
                unsafe { munmap(va as *mut _, size) };
                return Err(Error::AmdAllocFailed {
                    reason: "host-visible mmap failed (enable resizable BAR for VRAM, or check GTT availability)"
                        .into(),
                });
            }
            Some(unsafe { NonNull::new_unchecked(p as *mut u8) })
        } else {
            None
        };

        let mut gpu_id = self.node.gpu_id;
        let mut map_args = kfd::kfd_ioctl_map_memory_to_gpu_args {
            handle: mem_handle,
            device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
            n_devices: 1,
            n_success: 0,
        };
        let map_result = unsafe { ioctl::kfd_map_memory_to_gpu(self.kfd_fd.as_raw_fd(), &mut map_args as *mut _) };
        if let Err(e) = map_result {
            if map_args.n_success != 0 {
                self.unmap_kfd(mem_handle);
            }
            self.free_kfd(mem_handle);
            unsafe { munmap(va as *mut _, size) };
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_MAP_MEMORY_TO_GPU", errno: e as i32 });
        }
        if map_args.n_success != 1 {
            if map_args.n_success != 0 {
                self.unmap_kfd(mem_handle);
            }
            self.free_kfd(mem_handle);
            unsafe { munmap(va as *mut _, size) };
            return Err(Error::AmdAllocFailed {
                reason: format!("KFD map_memory_to_gpu reported {} success(es)", map_args.n_success),
            });
        }

        if zero_init {
            match host_ptr {
                Some(p) => unsafe { std::ptr::write_bytes(p.as_ptr(), 0, size) },
                // This seam has no copy queue, so it cannot zero a device-only
                // (non-host-mapped) buffer. Fail loud rather than silently return
                // garbage: `AmdAllocator::_alloc` passes `zero=false` here for
                // device-only buffers and zeroes them via SDMA itself, so a
                // `zero_init && !cpu_access` request reaching this point is a
                // caller bug (e.g. a future path that bypasses `_alloc`).
                None => {
                    self.unmap_kfd(mem_handle);
                    self.free_kfd(mem_handle);
                    unsafe { munmap(va as *mut _, size) };
                    return Err(Error::AmdAllocFailed {
                        reason: "zero_init requested for a device-only (non-host-mapped) allocation; \
                                 the caller must device-zero (e.g. via the SDMA copy queue)"
                            .into(),
                    });
                }
            }
        }

        self.va.insert(va as u64, size, mem_handle, tag);
        debug!(size, gpu_addr = va as u64, handle = mem_handle, tag = ?tag, "AmdAllocator alloc done");

        Ok(AllocResult { gpu_va: va as u64, host_ptr, handle: mem_handle, size })
    }

    fn free_raw(&self, gpu_va: u64, size: usize, handle: u64) {
        // Drop from the live registry into the freed-history ring *before* the
        // unmap, so a fault VA that lands here is classified as RECENTLY-FREED
        // (the use-after-free signal) rather than as a live allocation.
        self.va.remove(gpu_va);
        debug!(gpu_va, size, handle, "AmdIface free_raw");
        // 1. Unmap from GPU.
        let mut gpu_id = self.node.gpu_id;
        let mut unmap_args = kfd::kfd_ioctl_unmap_memory_from_gpu_args {
            handle,
            device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
            n_devices: 1,
            n_success: 0,
        };
        // SAFETY: fd is alive; handle is from a successful alloc.
        let _ = unsafe { ioctl::kfd_unmap_memory_from_gpu(self.kfd_fd.as_raw_fd(), &mut unmap_args as *mut _) };
        // 2. Drop host mapping (PROT_READ|PROT_WRITE for host-visible, or the
        //    PROT_NONE reservation for device-only). Both cases munmap the
        //    same VA region.
        // SAFETY: gpu_va is the VA returned by our own mmap.
        unsafe { munmap(gpu_va as *mut _, size) };
        // 3. Free the KFD allocation.
        let mut free_args = kfd::kfd_ioctl_free_memory_of_gpu_args { handle };
        // SAFETY: same as above.
        let _ = unsafe { ioctl::kfd_free_memory_of_gpu(self.kfd_fd.as_raw_fd(), &mut free_args as *mut _) };
    }

    fn setup_ring(&self, desc: &RingDesc) -> Result<QueueHandle> {
        let mut args = kfd::kfd_ioctl_create_queue_args {
            ring_base_address: desc.ring_gpu,
            write_pointer_address: desc.gart_gpu + desc.wptr_offset,
            read_pointer_address: desc.gart_gpu + desc.rptr_offset,
            doorbell_offset: 0,
            ring_size: desc.ring_size as u32,
            gpu_id: desc.gpu_id,
            queue_type: desc.queue_type,
            queue_percentage: kfd::KFD_MAX_QUEUE_PERCENTAGE,
            queue_priority: 7,
            ..Default::default()
        };
        if desc.eop_gpu != 0 {
            args.eop_buffer_address = desc.eop_gpu;
            args.eop_buffer_size = desc.eop_size;
        }
        if desc.ctx_gpu != 0 {
            args.ctx_save_restore_address = desc.ctx_gpu;
            args.ctx_save_restore_size = desc.ctx_save_restore_size;
            args.ctl_stack_size = desc.ctl_stack_size;
        }

        // SAFETY: kfd_fd is alive; args is well-typed.
        if let Err(e) = unsafe { ioctl::kfd_create_queue(self.kfd_fd.as_raw_fd(), &mut args as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_QUEUE", errno: e as i32 });
        }

        let (doorbell_base, doorbell) = match self.doorbell_mmap(args.doorbell_offset) {
            Ok(mapping) => mapping,
            Err(error) => {
                let mut destroy = kfd::kfd_ioctl_destroy_queue_args { queue_id: args.queue_id, ..Default::default() };
                // SAFETY: queue creation above succeeded and returned this id.
                if let Err(errno) = unsafe { ioctl::kfd_destroy_queue(self.kfd_fd.as_raw_fd(), &mut destroy as *mut _) }
                {
                    return Err(Error::AmdQueueStillActive {
                        queue_id: args.queue_id,
                        cause: format!("doorbell mapping failed ({error}); rollback destroy errno {errno}"),
                    });
                }
                return Err(error);
            }
        };
        debug!(queue_id = args.queue_id, doorbell_offset = args.doorbell_offset, "AMD queue created");
        Ok(QueueHandle { queue_id: args.queue_id, doorbell_base, doorbell })
    }

    fn update_queue_percentage(&self, queue_id: u32, ring_gpu: u64, ring_size: u32, percentage: u32) -> Result<()> {
        tracing::debug!(queue_id, percentage, "AMDKFD_IOC_UPDATE_QUEUE");
        let mut args = kfd::kfd_ioctl_update_queue_args {
            ring_base_address: ring_gpu,
            queue_id,
            ring_size,
            queue_percentage: percentage,
            queue_priority: 7,
        };
        // SAFETY: kfd_fd is alive; queue_id came from the matching create.
        unsafe { ioctl::kfd_update_queue(self.kfd_fd.as_raw_fd(), &mut args as *mut _) }
            .map_err(|errno| Error::AmdIoctl { ioctl: "AMDKFD_IOC_UPDATE_QUEUE", errno: errno as i32 })?;
        Ok(())
    }

    fn teardown_ring(&self, queue_id: u32, doorbell_base: NonNull<u8>) -> Result<QueueTeardown> {
        let mut args = kfd::kfd_ioctl_destroy_queue_args { queue_id, ..Default::default() };
        // SAFETY: `kfd_fd` is alive (held via Arc<OwnedFd>); the queue_id was
        // returned by KFD on the matching create_queue call.
        unsafe { ioctl::kfd_destroy_queue(self.kfd_fd.as_raw_fd(), &mut args as *mut _) }
            .map_err(|errno| Error::AmdIoctl { ioctl: "AMDKFD_IOC_DESTROY_QUEUE", errno: errno as i32 })?;
        // Release the per-queue doorbell MMIO page mapped in `doorbell_mmap`.
        // SAFETY: `doorbell_base` is the mmap base returned for this queue and
        // is no longer referenced once the queue is destroyed.
        if unsafe { munmap(doorbell_base.as_ptr().cast(), DOORBELL_PAGE_BYTES) } != 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            tracing::warn!(queue_id, errno, "teardown_ring: destroyed queue but leaked doorbell mapping");
            return Ok(QueueTeardown::DoorbellLeaked { errno });
        }
        Ok(QueueTeardown::Complete)
    }

    fn wait_events(&self, timeout_ms: u32, age: &mut u64) -> Result<Option<Error>> {
        let mut events: [kfd::kfd_event_data; 3] = [Default::default(); 3];
        events[0].event_id = self.queue_event_id;
        let aged = self.kfd_version >= (1, 14);
        if aged {
            events[0].__bindgen_anon_1.signal_event_data.last_event_age = *age;
        }
        events[1].event_id = self.mem_fault_event_id;
        events[2].event_id = self.hw_fault_event_id;
        let mut args = kfd::kfd_ioctl_wait_events_args {
            events_ptr: events.as_mut_ptr() as u64,
            num_events: events.len() as u32,
            wait_for_all: 0,
            timeout: timeout_ms,
            wait_result: 0,
        };
        // SAFETY: kfd_fd is alive; args + events live for the duration of the call.
        let rc = unsafe { ioctl::kfd_wait_events(self.kfd_fd.as_raw_fd(), &mut args as *mut _) };
        if let Err(e) = rc {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_WAIT_EVENTS", errno: e as i32 });
        }
        if aged {
            // SAFETY: slot [0] is the signal event, whose payload KFD writes back.
            *age = unsafe { events[0].__bindgen_anon_1.signal_event_data.last_event_age };
        }
        // Inspect each event's union payload. `gpu_id != 0` signals the fault
        // was actually written by KFD (the union is zero-initialized when
        // nothing fired). Slot [1] = memory fault, slot [2] = hw exception.
        // SAFETY: bindgen union access — we read whichever payload type
        // matches the event we registered.
        let mem = unsafe { events[1].__bindgen_anon_1.memory_exception_data };
        if mem.gpu_id != 0 {
            // Copy each field into a local before the format/tracing calls
            // below. The bindgen union payload is `#[repr(C)]` (naturally
            // aligned), so the by-value copies are for clarity, not to avoid an
            // unaligned reference.
            let gpu_id = mem.gpu_id;
            let va = { mem.va };
            let not_present = { mem.failure.NotPresent } != 0;
            let read_only = { mem.failure.ReadOnly } != 0;
            let no_execute = { mem.failure.NoExecute } != 0;
            let imprecise = { mem.failure.imprecise } != 0;
            let error_type = { mem.ErrorType };
            // Resolve the raw VA to its owning / stale / nearest allocation —
            // turns "fault at 0x7f…" into "fault +0x40 into a RECENTLY-FREED
            // scratch region", which is what actually localizes the bug.
            let class = self.va.classify(va);
            // Log at the fault site: the error that eventually surfaces this is
            // a delayed re-throw at the next `synchronize()`, far from here. Once
            // only — the memory-fault event is not auto-reset, so subsequent
            // `wait_events(0)` poll-fault calls re-observe the same fault.
            if !self.fault_logged.swap(true, Ordering::Relaxed) {
                tracing::error!(
                    gpu_id,
                    va = format!("{va:#x}").as_str(),
                    not_present,
                    read_only,
                    no_execute,
                    imprecise,
                    error_type,
                    classification = %class,
                    "AMD GPU memory fault"
                );
            }
            return Ok(Some(Error::GpuFault {
                gpu_id,
                va,
                not_present,
                read_only,
                no_execute,
                imprecise,
                error_type,
                class: class.to_string(),
            }));
        }
        let hw = unsafe { events[2].__bindgen_anon_1.hw_exception_data };
        if hw.gpu_id != 0 {
            let message = format!(
                "AMD GPU hardware exception on gpu_id={} reset_type={} reset_cause={} memory_lost={}",
                hw.gpu_id, hw.reset_type, hw.reset_cause, hw.memory_lost,
            );
            return Ok(Some(Error::Runtime { message }));
        }
        Ok(None)
    }
}

impl KfdIface {
    fn unmap_kfd(&self, handle: u64) {
        let mut gpu_id = self.node.gpu_id;
        let mut args = kfd::kfd_ioctl_unmap_memory_from_gpu_args {
            handle,
            device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
            n_devices: 1,
            n_success: 0,
        };
        let _ = unsafe { ioctl::kfd_unmap_memory_from_gpu(self.kfd_fd.as_raw_fd(), &mut args as *mut _) };
    }

    fn free_kfd(&self, handle: u64) {
        let mut args = kfd::kfd_ioctl_free_memory_of_gpu_args { handle };
        // SAFETY: self.kfd_fd is alive; handle is from a successful alloc.
        let _ = unsafe { ioctl::kfd_free_memory_of_gpu(self.kfd_fd.as_raw_fd(), &mut args as *mut _) };
    }

    /// Map a single doorbell window from `/dev/kfd` and return the mmap base
    /// (kept for a later `munmap`) plus the per-queue doorbell pointer. KFD
    /// doorbells are page-aligned regions of MMIO addresses; the per-queue
    /// doorbell address = page_base + (doorbell_offset & 0x1fff). The base is
    /// returned separately because `mmap` only guarantees 4 KiB alignment, so
    /// the page base is not recoverable from the offset pointer alone.
    fn doorbell_mmap(&self, doorbell_offset: u64) -> Result<(NonNull<u8>, NonNull<u64>)> {
        let page_base = doorbell_offset & !0x1fff;
        // SAFETY: standard libc::mmap; protections set for read+write MMIO.
        let p = unsafe {
            mmap(
                std::ptr::null_mut(),
                DOORBELL_PAGE_BYTES,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                self.kfd_fd.as_raw_fd(),
                page_base as i64,
            )
        };
        if p == libc::MAP_FAILED {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            return Err(Error::AmdIoctl { ioctl: "mmap(doorbell)", errno });
        }
        let base = NonNull::new(p as *mut u8).expect("doorbell page non-null");
        let offset_in_page = (doorbell_offset & 0x1fff) as usize;
        // SAFETY: offset_in_page < DOORBELL_PAGE_BYTES; alignment to u64 holds.
        let ptr = unsafe { base.as_ptr().add(offset_in_page) as *mut u64 };
        Ok((base, NonNull::new(ptr).expect("doorbell page non-null")))
    }
}

/// Compose the KFD `KFD_IOC_ALLOC_MEM_FLAGS_*` set for an allocation. This is
/// the ONLY place the flags are built; it reproduces the four pre-refactor flag
/// sets bit-for-bit (see the module doc / call sites).
fn compose_flags(kind: AllocKind, cpu_access: bool) -> u32 {
    match kind {
        AllocKind::DeviceVram { executable } => {
            let mut flags = kfd::KFD_IOC_ALLOC_MEM_FLAGS_VRAM
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
            if executable {
                flags |= kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;
            }
            if cpu_access {
                flags |= kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC;
            }
            flags
        }
        AllocKind::UncachedGtt => {
            kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED
        }
    }
}

/// Reserve `size` bytes of host VA so KFD can bind VRAM into it. A buffer of a
/// huge page or more starts on one: an anonymous `mmap` aligns only lengths
/// that are multiples of it (Linux ≥ 6.7, with THP on), and a VA off the
/// physical alignment takes no large PTE fragments (gfx942 GEMM over
/// 4128×4096×4096: 455 against 604 TF, 2026-11-06).
pub(crate) fn reserve_va(size: usize) -> Result<*mut libc::c_void> {
    let align = if size >= HUGE_PAGE { HUGE_PAGE } else { 1 };
    let span = size + (align - 1);
    // SAFETY: standard libc::mmap signature; no aliasing concerns at this point.
    let p = unsafe { mmap(std::ptr::null_mut(), span, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0) };
    if p == libc::MAP_FAILED {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        return Err(Error::AmdIoctl { ioctl: "mmap(VA reservation)", errno });
    }
    let (start, end) = (p as usize, p as usize + span);
    let va = start.next_multiple_of(align);
    let tail = (va + size).next_multiple_of(0x1000);
    // SAFETY: both ranges lie in the reservation above and nothing uses them.
    unsafe {
        if va > start {
            munmap(p, va - start);
        }
        if end > tail {
            munmap(tail as *mut _, end - tail);
        }
    }
    Ok(va as *mut _)
}

fn map_alloc_err(e: nix::errno::Errno, cpu_accessible: bool) -> Error {
    match e {
        nix::errno::Errno::ENOMEM => Error::AmdAllocFailed { reason: "ENOMEM (VRAM exhausted)".into() },
        nix::errno::Errno::EINVAL if cpu_accessible => {
            Error::AmdAllocFailed { reason: "EINVAL on host-visible VRAM alloc — enable resizable BAR".into() }
        }
        other => Error::AmdIoctl { ioctl: "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU", errno: other as i32 },
    }
}
