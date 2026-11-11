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
through 64 and defaults to 4. An atomic bitset tracks leases:

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
- AQL kernel packets carry no `completion_signal`. Vendor-IB PM4 waits/stores
  own timeline completion, with XCC0 `PRED_EXEC` on multi-XCC hardware.
- XCC0's timeline store can land while other XCCs still read the IB that holds
  it, so an AQL lane also counts retired submissions: the last packet of each
  publication names the lane's retirement signal as its AQL
  `completion_signal`, which the CP decrements once that packet retired on
  every XCC, raising the queue event. Kernarg slots, control programs, graph
  and linked-plan memory, and lane drains wait on that count. PM4 lanes retire
  on their timeline.

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

Every doorbell first runs a SeqCst fence, then flushes HDP from the host: on
every doorbell where kernargs and command buffers live in VRAM (multi-XCC
parts), elsewhere only when a dropped `HostWrite` guard deferred a flush that
no landed flush covers yet. Deferrals are counted, and a doorbell marks them
flushed only once its own flush has landed, so a doorbell that rings while
another is still flushing flushes too.

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
    fn wait_events(&self, timeout_ms: u32, age: &mut u64) -> Result<Option<Error>>;

    // Defaulted hooks. `KfdIface` overrides all but `publication_checkpoint`,
    // which only the test mock overrides.
    fn queue_event_mailbox(&self) -> Option<QueueEventMailbox> {
        None
    }
    fn publication_checkpoint(&self, _stage: PublicationStage) -> Result<()> {
        Ok(())
    }
    fn update_queue_percentage(&self, _queue_id: u32, _ring_gpu: u64, _ring_size: u32, _percentage: u32) -> Result<()> {
        Ok(())
    }
    fn flush_hdp(&self) {}
}
```

Ring, GART, EOP and context-save buffers, and each compute queue's error
words, are allocated above this seam. `setup_ring` activates those resources
and maps the doorbell. `update_queue_percentage` is what re-maps an AQL queue
so CP firmware re-reads its cached `amd_queue_t` scratch descriptor — an unmap
and a remap at lane creation and whenever scratch grows; a failed remap
poisons the device. Each remap preempts every queue of the process on the
device, so a grow at least doubles the per-thread scratch size (or takes the
request when that is larger), up to the largest size
`COMPUTE_TMPRING_SIZE.WAVESIZE` describes. When `queue_event_mailbox` is
present, every completion store carries a KFD event-interrupt companion so
`wait_events` wakes without polling.

A compute queue's error words are the AQL `queue_inactive_signal`, where the CP
writes a code as it halts the queue, and the `ErrorReason` word its CWSR
headers point at, where KFD sets the bits of a queue exception. Both raise the
queue event. A timeline wait reads every lane's words when it wakes from
`wait_events` and when it times out, and poisons the device with
`Error::AmdQueueError`, which names the decoded cause (see
[Debugging](./debugging.md)).

`flush_hdp` flushes the host's VRAM stores out of HDP and waits for the flush
to land. `KfdIface` writes KFD's MMIO-remap flush register or, where KFD gives
no remap page (an SR-IOV VF), stores to a host-visible VRAM word and reads it
back.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `SVOD_DEVICE` | `CPU` (`METAL:0` on macOS) | Select default tensor device, for example `AMD:0` (`HIP` is an alias) |
| `SVOD_AMD_BACKEND` | `kfd` | AMD backend; only `kfd` is currently accepted |
| `SVOD_AMD_HW_QUEUES` | 4 | Bounded compute-lane count, clamped to 1 through 64 |
| `SVOD_AMD_AQL` | unset | Any value other than `0` forces AQL on single-XCC hardware; read at each queue creation |
| `SVOD_AMD_SCOPED_SYNC` | unset | `=0` replaces every storage-scoped host wait with a full device drain and stops producer recording; latched once per process |
| `SVOD_PM4_GRAPH` | unset | `=1` enables PM4 graph capture; only `1` counts; read once at device creation |
| `AMD_DISABLE_SDMA` | unset | Set to anything to skip the SDMA copy queue, forcing host-visible buffers |
| `SVOD_KFD_TOPOLOGY` | sysfs | Override KFD topology root for tests |
| `SVOD_DEBUG_DISPATCH` | unset | Set to anything to print each program's load descriptor and each dispatch's grid, scratch and buffer addresses |
| `SVOD_DUMP_AMD_IR` | unset | Directory receiving each kernel's AMD LLVM IR as `<mcpu>_<module>_<digest>.ll` |
| `SVOD_DUMP_AMD_ASM` | unset | Directory receiving each kernel's AMDGCN assembly as `<mcpu>_<module>_<digest>.s`, the stem of its `.ll` (one more clang run per kernel) |
| `SVOD_OBJECT_CACHE`, `SVOD_OBJECT_CACHE_DIR`, `SVOD_OBJECT_CACHE_MAX_BYTES` | on | The shared on-disk object cache, see the [CPU page](../cpu.md) |
| `SVOD_AM_DEBUG` | unset | AM bring-up only: read registers back after writing them |
| `SVOD_AM_MCBASE` | unset | AM bring-up only: `raw`, `fb`, or `fbxgmi` MC aperture base |

There is no `SVOD_AMD_SINGLE_QUEUE`. Set `SVOD_AMD_HW_QUEUES=1` when a single
hardware lane is required.
