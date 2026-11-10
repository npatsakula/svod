//! AMD GPU support for the device crate (KFD-direct, no HIP runtime).
//!
//! Compiled on all Unix hosts (the FFI is `cfg(unix)` — `nix`/KFD ioctls).
//! Availability is decided at runtime, not compile time: the device registry
//! registers the AMD factory only when [`topology::has_devices`] finds a
//! supported GPU, so a host with no `/dev/kfd` simply never reaches this code.
//!
//! # Hardware findings and open gates
//!
//! The HCQ port is pinned to Tinygrad `8c8b43de62515abe6c820b1de5aa26b30f48e43a`.
//! What the code cannot express on its own:
//!
//! - **Several user queues must not share an in-queue HDP flush.** Every
//!   queue once flushed HDP with a `WAIT_REG_MEM` handshake on one
//!   GPU_HDP_FLUSH_REQ/DONE bit (the kernel gives each ring its own; user
//!   queues cannot know theirs). Concurrent queues stole one another's DONE
//!   and spun forever: gfx1151 wedged MES (`MES failed to respond to
//!   msg=MISC (WAIT_REG_MEM)`, then every `REMOVE_QUEUE`) into an
//!   unrecoverable GPU reset, and gfx942 stalled its timelines. The host
//!   flushes HDP through the KFD MMIO remap page instead, as ROCr does, and
//!   four lanes pass the tk suite on gfx942, gfx1201 and gfx1151
//!   (2026-11-10). Still true: gfx11 MEC firmware rejects the AMD
//!   barrier-value vendor AQL packet as an illegal opcode (probed), and
//!   barrier-AND is decrement-to-zero only, so no AQL packet there expresses
//!   a monotonic timeline wait.
//! - **Validated on gfx1151**: PM4 single lane through the full tensor, ONNX,
//!   and model suites in parallel; forced AQL through the tensor suite
//!   single-threaded. **Pending hardware gates**: multi-XCC AQL with XCC0
//!   `PRED_EXEC` predication, two-GPU peer/sentinel validation, and
//!   near-capacity ring stress.
//! - **Open on gfx1151**: the merged tensor doctest binary under
//!   `SVOD_DEVICE=AMD:0` faults (`NotPresent` at offset 0 of a live
//!   `QueueGart` descriptor page) in a handful of random doctests per run,
//!   and passes with `--test-threads=1`. The unit suites do not show it.
//!   Reproduced on `7f7f1f5f` (2026-09-04). Same symptom as the root PD clear
//!   race fixed in `AmdIface::open` on gfx942; not re-tested on gfx1151.
//! - **Structural debt**: linked plans publish SDMA work on a plan-local
//!   timeline, so `AmdCopyQueue` must retain foreign finalizers; the wider
//!   fix is one per-device timeline signal as in Tinygrad's `HCQCompiled`.
//!   `AmdCopyQueue::drop` is unreachable today because the installed
//!   `SignalPool` holds the `AmdDeviceCore` while the core holds the pool.
//!   Cross-queue `opt_deps` wait narrowing in graph capture is deferred.
//! - AMD exposes no verified per-allocation peer-access query, so mixed-device
//!   plans stage through the host unless a target reports both directions.

pub mod sys;
pub mod topology;

/// Userspace AMD ("AM") driver — groundwork for a second [`iface::AmdIface`]
/// backend that drives the GPU's PCI BARs directly, bypassing the kernel
/// amdgpu/KFD driver. Not wired into [`AmdDevice`] yet (`SVOD_AMD_BACKEND`
/// accepts only `kfd`). The memory-manager submodules (`am::mm`) are pure
/// logic and unit-test without a GPU; the privileged bring-up targets the
/// gfx9.4.x SR-IOV VF. Compiled on all Unix hosts (no extra deps) so it's
/// always type-checked, linted, and tested.
#[cfg(unix)]
pub mod am;

#[cfg(unix)]
pub mod allocator;
#[cfg(unix)]
pub mod connector;
#[cfg(unix)]
pub mod device;
#[cfg(unix)]
pub mod graph;
#[cfg(unix)]
pub mod iface;
#[cfg(unix)]
pub mod kernarg;
#[cfg(unix)]
pub(crate) mod linked_plan;
#[cfg(test)]
pub(crate) use linked_plan::AmdLinkedPlan;
pub mod occupancy;
#[cfg(unix)]
pub mod pmc;
#[cfg(unix)]
pub mod program;
#[cfg(unix)]
pub mod queue;
#[cfg(unix)]
pub mod signal;
#[cfg(unix)]
pub mod va_registry;

#[cfg(unix)]
pub use allocator::AmdAllocator;
#[cfg(unix)]
pub use device::{AmdDevice, AmdDeviceCore};
#[cfg(unix)]
pub use graph::AmdGraph;
#[cfg(unix)]
pub use iface::{AllocKind, AllocResult, AmdIface, KfdIface, QueueHandle, QueueTeardown, RingDesc};
#[cfg(unix)]
pub use kernarg::KernargArena;
#[cfg(unix)]
pub use program::AmdProgram;
#[cfg(unix)]
pub use queue::{AmdComputeQueue, AmdCopyQueue};
#[cfg(unix)]
pub use signal::{AmdSignal, SignalPool, Timeline};
pub use topology::{AmdNode, enumerate, has_devices};
#[cfg(unix)]
pub use va_registry::AllocTag;
