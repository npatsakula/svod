---
sidebar_label: Queues & Dispatch
---

# Queues & Dispatch

The AMD backend preserves Tinygrad's validated PM4, AQL, and SDMA packet
semantics, but uses Rust ownership for queue scheduling and failure handling.
The central rule is simple: **one non-clone lease is the only publication
authority for a compute lane**.

## Compute lanes

`AmdDeviceCore` owns a bounded `QueuePool`. Its slots are fixed `OnceLock`s and
queues are created lazily up to `SVOD_AMD_HW_QUEUES`, which is clamped to 1
through 64 and defaults to 4 on multi-XCC CDNA, 1 everywhere else. An atomic
bitset tracks leases:

- claiming an initialized idle lane is an atomic compare-exchange;
- queue creation is a cold serialized path;
- when every lane is leased, callers park on a condition variable;
- dropping `QueueLease` clears the bit and wakes one waiter;
- queues never co-tenant host publishers.

Programs and graph templates never hold a `QueueLease`. `OwnerCtx` holds the
logical plan state — the device core and allocator, the newest completion,
the PMC selection, an optional linked replay template — and a `session` slot
that only the direct-dispatch fallback fills with its epoch lease.

A thread parks at most 30 s per wait; two consecutive expiries return
`Error::TimelineTimeout` for "AMD lane acquisition", and a poisoned device
wakes every waiter with its poison error.

Direct semantic fallback keeps one lease across all kernels in a replay epoch,
then `PlanContext::finish_replay` releases it. A later epoch waits the prior
finalizer before acquiring another lane, because a different queue would not
inherit the old queue's FIFO ordering. Graph and native linked replay already
wait before reusing their mutable kernarg/control storage and lease a lane for
each publication epoch.

## Native queues

`AmdComputeQueue` owns a 16 MiB host-visible ring, GART read/write pointers, a
doorbell mapping, and KFD queue backing. Packet format is selected once:

```text
PM4 = num_xcc == 1 && SVOD_AMD_AQL is unset or "0"
AQL = otherwise
```

- PM4 queues publish raw dwords and ring the next dword index.
- AQL queues publish 64-byte packets and ring the index of the last packet
  written.
- AQL kernel `completion_signal` remains zero. Vendor-IB PM4 waits/stores own
  timeline completion, with XCC0 `PRED_EXEC` on multi-XCC hardware.

The lane lease eliminates compute co-tenancy. `AmdComputeQueue.inner` still uses
a mutex as a Rust aliasing guard; it is uncontended on the normal compute path.
The singleton SDMA queue is independently mutex-protected because copies from
different plans may share it.

## Publication

Native linked replay splits a submission into preparation and publication:

1. Validate program identity, concrete buffer ownership, ABI, launch geometry,
   patch tables, and hardware stream limits.
2. Reserve and write kernargs/control data.
3. Acquire ring headroom.
4. Register a prepared finalizer when device-wide drains need to observe a
   plan-owned timeline.
5. Publish packets and doorbells.
6. Mark the finalizer published.

If an error unwinds after registration, the prepared finalizer becomes failed.
A concurrent drain wakes and fails immediately rather than waiting for a
terminal store that was never published. The physical device is then poisoned,
so the lane cannot be reused and hardware-referenced allocations are
quarantined.

Ordinary per-kernel dispatch (`submit_hcq_dispatch`) is simpler: its finalizer
is born published after the doorbell and the caller registers it in flight; a
failure before the doorbell rolls the ring and the timeline reservation back
instead.

PM4, AQL, and SDMA publication all check the KFD read pointer before wrapping
their rings (AQL and SDMA read a monotonic value; the PM4 pointer is
queue-relative and its epoch is rebuilt from the producer index). Ordinary
dispatch additionally bounds in-flight timeline values. PM4 timeline values
drain and reset at the 2^31 watermark because hardware wait/store packets
compare the low 32 bits.

## Resource lifetime

Every direct submission finalizer retains its code object. Graphs and linked
plans retain all code objects they link. Persistent kernarg, resident command,
control, timestamp, and PMC allocations remain owned until their exact replay
completion is retired.

Queue lifecycle is explicit:

```text
Constructing -> Active
Constructing -> Destroyed | Quarantined
Active -> Destroyed
Active -> Quarantined
```

Orderly compute teardown is drain, KFD `DESTROY_QUEUE`, scratch release, then
ring/GART/context release. A failed drain or destroy poisons the physical device
and leaves all potentially referenced backing mapped. Doorbell unmap failure
after successful queue destruction is reported as a host mapping leak, but does
not unnecessarily quarantine safe GPU backing.

If `CREATE_QUEUE` succeeds but doorbell mapping and rollback destruction both
fail, `setup_ring` returns `AmdQueueStillActive`. The caller poisons the device
before allocation guards unwind, preventing a live KFD queue from observing
freed ring memory.

A panic that unwinds through a lane quarantines that lane but does not poison
the device; only a reservation abandoned *after* its doorbell rang
(`TimelineReservation`, `PreparedPublication`) poisons, whether the cause was
an error or a panic. Signal slots return to the pool unconditionally unless the
device is poisoned, in which case they are withheld because an abandoned queue
may still target them.

## Device-wide drains

Each lane owns a queue timeline and a FIFO of non-queue finalizers. The device
core keeps weak references to every initialized lane. `synchronize_all`
snapshots those lanes and waits their timelines without taking publication
locks. Host reads, writes, and destructive frees prefer the scoped
`wait_storage`, which waits only the submissions recorded against that storage
base and falls back to the full drain for an unknown VA or under
`SVOD_AMD_SCOPED_SYNC=0`.

Native replay additionally re-validates every operation before republishing: a
PROGRAM must still be an `AmdProgram` whose core is the very same `Arc`
(`Arc::ptr_eq`, not an allocator merely reporting `AMD:N`) with unchanged PM4
and AQL program addresses, and a COPY lane requires an installed SDMA queue.

## Backend seam

KFD operations are isolated behind `AmdIface`:

```rust
pub trait AmdIface: Send + Sync + std::fmt::Debug {
    fn alloc_raw(
        &self,
        size: usize,
        kind: AllocKind,
        tag: AllocTag,
        cpu_access: bool,
        zero: bool,
    ) -> Result<AllocResult>;
    fn free_raw(&self, gpu_va: u64, size: usize, handle: u64);
    fn setup_ring(&self, desc: &RingDesc) -> Result<QueueHandle>;
    fn teardown_ring(&self, queue_id: u32, doorbell_base: NonNull<u8>) -> Result<QueueTeardown>;
    fn wait_events(&self, timeout_ms: u32) -> Result<Option<Error>>;

    // Defaulted hooks. `KfdIface` overrides the first and the last; the
    // test mock overrides `publication_checkpoint`.
    fn queue_event_mailbox(&self) -> Option<QueueEventMailbox> {
        None
    }
    fn publication_checkpoint(&self, _stage: PublicationStage) -> Result<()> {
        Ok(())
    }
    fn update_queue_percentage(&self, _queue_id: u32, _ring_gpu: u64, _ring_size: u32, _percentage: u32) -> Result<()> {
        Ok(())
    }
}
```

Ring, GART, EOP, context-save, and inactive-signal buffers are allocated above
this seam. `setup_ring` activates those resources and maps the doorbell.
`update_queue_percentage` is what re-maps an AQL queue so CP firmware re-reads
its cached `amd_queue_t` scratch descriptor — at lane creation and whenever
scratch grows; a failed remap poisons the device. When `queue_event_mailbox`
is present, every completion store carries a KFD event-interrupt companion so
`wait_events` wakes without polling.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `SVOD_DEVICE` | `CPU` (`METAL:0` on macOS) | Select default tensor device, for example `AMD:0` (`HIP` is an alias) |
| `SVOD_AMD_BACKEND` | `kfd` | AMD backend; only `kfd` is currently accepted |
| `SVOD_AMD_HW_QUEUES` | 4 on multi-XCC, else 1 | Bounded compute-lane count, clamped to 1 through 64 |
| `SVOD_AMD_AQL` | unset | Any value other than `0` forces AQL on single-XCC hardware; read at each queue creation |
| `SVOD_AMD_SCOPED_SYNC` | unset | `=0` replaces every storage-scoped host wait with a full device drain and stops producer recording; latched once per process |
| `SVOD_PM4_GRAPH` | unset | `=1` enables PM4 graph capture; only `1` counts; read once at device creation |
| `AMD_DISABLE_SDMA` | unset | Set to anything to skip the SDMA copy queue, forcing host-visible buffers |
| `SVOD_KFD_TOPOLOGY` | sysfs | Override KFD topology root for tests |
| `SVOD_DEBUG_DISPATCH` | unset | Set to anything to print program-load and dispatch grid, kernarg, scratch, and buffer addresses |
| `SVOD_DUMP_AMD_IR` | unset | Directory receiving each kernel's AMD LLVM IR as `<mcpu>_<module>.ll` |
| `SVOD_OBJECT_CACHE`, `SVOD_OBJECT_CACHE_DIR`, `SVOD_OBJECT_CACHE_MAX_BYTES` | on | The shared on-disk object cache, see the [CPU page](../cpu.md) |
| `SVOD_AM_DEBUG` | unset | AM bring-up only: read registers back after writing them |
| `SVOD_AM_MCBASE` | unset | AM bring-up only: `raw`, `fb`, or `fbxgmi` MC aperture base |

There is no `SVOD_AMD_SINGLE_QUEUE`. Set `SVOD_AMD_HW_QUEUES=1` when a single
hardware lane is required.
