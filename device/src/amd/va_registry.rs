//! VA → allocation registry: a diagnostic side-table mapping every live GPU
//! virtual address range back to its owning allocation.
//!
//! The KFD fault path reports only a raw faulting VA (`NotPresent=1` etc.) with
//! no way to answer the question that actually localizes the bug: *what was at
//! that address?* This registry closes that gap. [`KfdIface::alloc_raw`] records
//! each mapped range; [`KfdIface::free_raw`] removes it (and retains it in a
//! bounded freed-history ring for use-after-free triage); on a fault,
//! [`KfdIface::wait_events`] calls `VaRegistry::classify` to enrich the fault
//! message — "this VA is +0x40 into a LIVE scratch alloc", "this VA is in a
//! RECENTLY-FREED region (stale/use-after-free)", or "this VA is in no tracked
//! allocation; nearest live neighbours are …".
//!
//! Pure bookkeeping with no GPU dependency, so the classification logic is
//! unit-/property-testable on any host.
//!
//! [`KfdIface::alloc_raw`]: crate::amd::iface::KfdIface
//! [`KfdIface::free_raw`]: crate::amd::iface::KfdIface
//! [`KfdIface::wait_events`]: crate::amd::iface::KfdIface

use std::collections::{BTreeMap, VecDeque};
use std::ops::Bound;

use parking_lot::Mutex;

/// What an allocation is for. Coarse on purpose — derived at the two
/// `alloc_raw` call sites. The distinction that matters for fault triage is
/// **scratch vs everything else**: scratch is the only shared, GPU-only,
/// dynamically realloc'd-and-freed region, and the historical `NotPresent`
/// culprit (see `amd-dispatch-lock-invariant`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocTag {
    /// General device VRAM (tensor data, code, kernargs, EOP / ctx-save).
    Vram,
    /// GTT-pinned host-visible control memory (queue ring, GART, signal slots).
    Gtt,
    SignalPool,
    QueueRing,
    QueueGart,
    QueueInactive,
    Staging,
    Kernarg,
    /// Register-spill scratch — GPU-only VRAM, realloc'd per kernel as private
    /// segment sizes grow.
    Scratch,
}

impl AllocTag {
    pub(crate) fn label(self) -> &'static str {
        match self {
            AllocTag::Vram => "VRAM buffer",
            AllocTag::Gtt => "GTT control",
            AllocTag::SignalPool => "GTT signal pool",
            AllocTag::QueueRing => "GTT queue ring",
            AllocTag::QueueGart => "GTT queue GART",
            AllocTag::QueueInactive => "GTT queue-inactive signal",
            AllocTag::Staging => "GTT SDMA staging",
            AllocTag::Kernarg => "kernarg",
            AllocTag::Scratch => "scratch",
        }
    }
}

/// One live allocation's bookkeeping (keyed by its base VA in the registry).
#[derive(Clone, Copy, Debug)]
struct Record {
    size: usize,
    handle: u64,
    tag: AllocTag,
}

/// A freed allocation retained for use-after-free triage.
#[derive(Clone, Copy, Debug)]
struct Freed {
    base: u64,
    size: usize,
    handle: u64,
    tag: AllocTag,
}

/// How many recently-freed regions to retain. Bounded so the table can't grow
/// without limit; faults consult it linearly, but only on the (rare, terminal)
/// fault path. Large enough to span the realloc churn of a model load.
pub(crate) const FREED_HISTORY: usize = 256;

/// A nearest-neighbour live allocation, with the gap from the queried VA to its
/// closer edge — for overrun / wild-pointer triage when a VA hits no region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Neighbor {
    pub base: u64,
    pub end: u64,
    pub tag: AllocTag,
    /// Distance from the queried VA to this allocation's nearer edge: bytes
    /// past `end` for a below-neighbour, bytes before `base` for an above one.
    pub gap: u64,
}

/// Where a queried VA falls relative to the tracked allocations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum VaClass {
    /// Inside a currently-mapped allocation.
    Live { base: u64, end: u64, offset: u64, handle: u64, tag: AllocTag },
    /// Inside a region that was freed/unmapped — stale VA / use-after-free.
    Freed { base: u64, end: u64, handle: u64, tag: AllocTag },
    /// In no tracked region. Carries the nearest live neighbours (if any).
    Unmapped { below: Option<Neighbor>, above: Option<Neighbor> },
}

impl std::fmt::Display for VaClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VaClass::Live { base, end, offset, handle, tag } => write!(
                f,
                "va is at offset +{offset:#x} within a LIVE {} allocation [{base:#x}, {end:#x}) (handle={handle:#x})",
                tag.label(),
            ),
            VaClass::Freed { base, end, handle, tag } => write!(
                f,
                "va is within a RECENTLY-FREED {} region [{base:#x}, {end:#x}) (handle={handle:#x}) — \
                 use-after-free: a stale/recycled VA still referenced by an in-flight kernel",
                tag.label(),
            ),
            VaClass::Unmapped { below, above } => {
                write!(f, "va is in NO tracked allocation")?;
                match below {
                    Some(n) => write!(
                        f,
                        "; nearest live below: {} [{:#x}, {:#x}) (va is +{:#x} past its end)",
                        n.tag.label(),
                        n.base,
                        n.end,
                        n.gap,
                    )?,
                    None => write!(f, "; no live allocation below")?,
                }
                match above {
                    Some(n) => write!(
                        f,
                        "; nearest live above: {} [{:#x}, {:#x}) (va is {:#x} before its start)",
                        n.tag.label(),
                        n.base,
                        n.end,
                        n.gap,
                    ),
                    None => write!(f, "; no live allocation above"),
                }
            }
        }
    }
}

#[derive(Debug, Default)]
struct Inner {
    /// Live allocations keyed by base VA (sorted → range queries for the
    /// containing / nearest-neighbour lookups).
    live: BTreeMap<u64, Record>,
    /// Most-recently-freed at the front, bounded to [`FREED_HISTORY`].
    freed: VecDeque<Freed>,
}

/// Per-device VA registry. One per [`KfdIface`](crate::amd::iface::KfdIface);
/// a fault corrupts the whole VM, so per-device is the right granularity.
#[derive(Debug, Default)]
pub(crate) struct VaRegistry {
    inner: Mutex<Inner>,
}

impl VaRegistry {
    /// Record a freshly-mapped allocation. Called from `alloc_raw` after the
    /// `MAP_MEMORY_TO_GPU` ioctl succeeds.
    pub(crate) fn insert(&self, base: u64, size: usize, handle: u64, tag: AllocTag) {
        self.inner.lock().live.insert(base, Record { size, handle, tag });
    }

    /// Drop a live allocation and retain it in the freed-history ring. Called
    /// from `free_raw`. A `base` not in the live map (double-free, or a VA that
    /// was never registered — e.g. the process-global event page) is ignored.
    pub(crate) fn remove(&self, base: u64) {
        let mut inner = self.inner.lock();
        if let Some(r) = inner.live.remove(&base) {
            inner.freed.push_front(Freed { base, size: r.size, handle: r.handle, tag: r.tag });
            inner.freed.truncate(FREED_HISTORY);
        }
    }

    /// Resolve a faulting VA to its owning (or most-recently-owning, or nearest)
    /// allocation. Live takes precedence over freed (a re-allocated VA reads as
    /// `Live`, not as a stale `Freed`).
    pub(crate) fn classify(&self, va: u64) -> VaClass {
        let inner = self.inner.lock();

        // 1. Inside a live allocation? The greatest base <= va is the only
        //    candidate that can contain va.
        if let Some((&base, rec)) = inner.live.range(..=va).next_back() {
            let end = base.saturating_add(rec.size as u64);
            if va < end {
                return VaClass::Live { base, end, offset: va - base, handle: rec.handle, tag: rec.tag };
            }
        }

        // 2. Inside a recently-freed region? Front-to-back = newest-first, so
        //    the first containing match is the most recent occupant.
        for fr in &inner.freed {
            let end = fr.base.saturating_add(fr.size as u64);
            if fr.base <= va && va < end {
                return VaClass::Freed { base: fr.base, end, handle: fr.handle, tag: fr.tag };
            }
        }

        // 3. Wild / overrun: report the nearest live neighbours either side.
        let below = inner.live.range(..=va).next_back().map(|(&base, rec)| {
            let end = base.saturating_add(rec.size as u64);
            Neighbor { base, end, tag: rec.tag, gap: va.saturating_sub(end) }
        });
        let above = inner.live.range((Bound::Excluded(va), Bound::Unbounded)).next().map(|(&base, rec)| {
            let end = base.saturating_add(rec.size as u64);
            Neighbor { base, end, tag: rec.tag, gap: base.saturating_sub(va) }
        });
        VaClass::Unmapped { below, above }
    }
}
