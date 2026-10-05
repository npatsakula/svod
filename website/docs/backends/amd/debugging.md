---
sidebar_label: Debugging
---

# Debugging & Fault Triage

When a GPU kernel touches memory it shouldn't, KFD reports a fault with a raw
virtual address and little else. This page covers the tools the backend has for
turning that into a diagnosis: the VA→allocation registry that classifies the
faulting address, the poison latch that stops the device cleanly, and the
dispatch/tracing instrumentation.

---

## The problem: a bare faulting VA

A `WAIT_EVENTS` memory-fault event hands back a `kfd_hsa_memory_exception_data`
with the faulting `va`, failure flags (`NotPresent`, `ReadOnly`, `NoExecute`,
`imprecise`), and an `ErrorType`. That tells you *where* the GPU faulted but not
*what was there* — which is the question that actually localizes the bug. The
panic that eventually surfaces it is a delayed re-throw at the next
`synchronize()`, far from the fault site.

---

## The VA registry

`device/src/amd/va_registry.rs` is a diagnostic side-table that maps every live
GPU VA range back to its owning allocation. It is pure bookkeeping — no GPU
dependency — so the classification logic is unit- and property-tested on any
host. One `VaRegistry` lives on `KfdIface` (a fault corrupts the whole VM, so
per-device is the right granularity).

It is maintained at the two ends of an allocation's life:

- **`alloc_raw`** calls `va.insert(base, size, handle, tag)` after
  `MAP_MEMORY_TO_GPU` succeeds.
- **`free_raw`** calls `va.remove(base)` *before* the unmap — so a fault that
  lands on a just-freed VA classifies as use-after-free rather than as a live
  allocation.

### Tags

Each allocation is tagged with its purpose (`AllocTag`). `Vram` and `Gtt` are
the defaults derived from the `AllocKind`; the finer tags are passed explicitly
by the `alloc_*_tagged` call sites:

| Tag | Covers |
|---|---|
| `Vram` | General device VRAM — tensor data, code objects, EOP/ctx-save |
| `Gtt` | Host-visible control memory — GTT, and the graph / linked-plan IB and control buffers |
| `Kernarg` | Kernarg arenas (host-visible VRAM) — the per-dispatch, graph, and linked-plan argument pages |
| `SignalPool` | The GTT signal-slot pool |
| `QueueRing` / `QueueGart` / `QueueInactive` | A queue's ring, GART page, and queue-inactive signal |
| `Staging` | The GTT SDMA bounce buffer |
| `Scratch` | Register-spill scratch — GPU-only VRAM, realloc'd per kernel |

The distinction that matters is **scratch vs everything else**: scratch is the
only shared, GPU-only, dynamically realloc'd-and-freed region, and the historical
`NotPresent` culprit. The per-process KFD event page is allocated outside
`alloc_raw` and is not tracked: a fault there classifies as unmapped.

### Classification

The registry keeps a `BTreeMap` of live ranges (keyed by base VA, for range
queries) plus a bounded ring of the **256** most-recently-freed regions
(`FREED_HISTORY`). `classify(va)` resolves a faulting address with this
precedence:

```text
1. Live    — va is inside a currently-mapped allocation
             (live takes precedence, so a re-allocated VA reads Live, not stale)
2. Freed   — va is inside a recently-freed region → use-after-free
3. Unmapped — va is in no tracked region; report nearest live neighbours + gaps
```

The `Display` rendering is what lands in the fault message:

```text
Live:     va is at offset +0x40 within a LIVE scratch allocation
          [0x7f…000, 0x7f…400) (handle=0x42)

Freed:    va is within a RECENTLY-FREED scratch region [0x…, 0x…) (handle=0x…)
          — use-after-free: a stale/recycled VA still referenced by an
          in-flight kernel

Unmapped: va is in NO tracked allocation; nearest live below: VRAM buffer
          [0x…, 0x…) (va is +0x80 past its end); nearest live above: …
```

---

## How a fault is reported

In `KfdIface::wait_events` (`device/src/amd/iface.rs`), when the memory-fault
event has fired (`gpu_id != 0`), the fields are copied out of the bindgen union
payload into locals, the VA is classified, and an enriched message is built:

```text
AMD GPU memory fault on gpu_id=… va=0x… (NotPresent=true ReadOnly=false NoExecute=false
Imprecise=false ErrorType=…) — va is at offset +0x40 within a LIVE scratch …
```

It is logged **once** via a `fault_logged: AtomicBool` latch and a
`tracing::error!`. The one-shot matters: the memory-fault event is not
auto-reset, so subsequent poll-fault calls (`wait_events(0)`) re-observe the same
fault — logging every time would spam. It is then returned as a typed
`Error::GpuFault`, whose `Display` is the string above; the poison latch
records that text and re-throws it as an `Error::Runtime` (prefixed
`runtime error: `) at every later entry point. A wait that expires without a
fault is a distinct `Error::TimelineTimeout`.
(A hardware-exception event, slot `[2]`, reports
`reset_type`/`reset_cause`/`memory_lost` instead — those have no faulting VA to
classify.)

---

## The poison latch

A memory fault corrupts the whole per-VM page table, so the device is dead after
one. `AmdDeviceCore` (`device/src/amd/device.rs`) holds a poison latch —
`poisoned: AtomicBool` + `error_msg: OnceLock<String>` — checked at every
dispatch and synchronize entry point:

- `poison(msg)` records the message once and sets the flag;
- `is_poisoned()` is the hot-path gate;
- `poison_error()` returns the recorded `Error::Runtime` if poisoned;
- `poll_faults_nonblocking()` issues `wait_events(0)` when a signal wait
  reaches its deadline (30 s for dispatch, copy and lane acquisition), so the
  real error is attached to the timeout rather than a bare deadline. (The
  spin-escalation path also breaks out early on a fault, but through a short
  *blocking* `wait_events` instead of this poll.) `poison()` also wakes every
  thread parked on the lane pool.

Once poisoned, every `synchronize`/`execute` against any lane on the device
fails fast — the GPU state and cached mappings are no longer trustworthy.

---

## Dispatch instrumentation: `SVOD_DEBUG_DISPATCH`

Setting `SVOD_DEBUG_DISPATCH` (to anything) turns on `eprintln` dumps at two
points, both in `device/src/amd/program.rs`:

- **`[program-load]`** — per program: kernarg/private/group sizes,
  `kernel_code_properties` (decoded bit-by-bit), user-SGPR count, `wave32`, and
  the raw `rsrc1/2/3`. It flags `kernel_code_properties` bits that the loader
  does *not* populate (which would make the kernel read garbage pointers and
  fault).
- **`[dispatch tv=…]`** — per dispatch: kernel name, `grid`, `local`, `is_pm4`,
  the kernarg GPU VA, the scratch VA, and each buffer's VA.

This is the fastest way to see exactly which VAs a faulting dispatch touched, to
cross-reference against the registry's classification.

---

## Tracing setup (`RUST_LOG`)

The backend uses the `tracing` crate (`debug!`, `tracing::error!`) but installs
**no subscriber** — that's the host binary's job. The `alloc_raw`/`free_raw`
`debug!` lines and the one-shot fault `error!` only appear if a subscriber is
installed and the level allows it.

The example binaries that install one call `tracing_subscriber::fmt::init()` in
`main` (it honours `RUST_LOG`):

```bash
# Surface the alloc/free debug lines and the fault error from gigaam_infer:
RUST_LOG=svod_device=debug \
SVOD_DEVICE=AMD:0 \
  cargo run --release -p svod-model --example gigaam_infer -- ./audio.wav
```

:::tip[The pipeline debugger]
For *compiler*-side issues (IR extraction, LLVM IR, UOp trees) rather than the
driver, the project ships a `/svod-debug` skill documenting the frontend →
codegen tracing targets (`SVOD_DUMP_LLVM_IR`, `SVOD_DUMP_AMD_IR`, per-stage
`RUST_LOG` targets, `setup_test_tracing()`). That is a separate toolkit from the
driver-side fault triage on this page.
:::

---

## A worked triage

When a `NotPresent` fault recurs, the workflow is:

1. The fault message already names the class — read it first. "LIVE scratch"
   points at the scratch realloc path; "RECENTLY-FREED" is a use-after-free of a
   buffer freed while a kernel still referenced it; "NO tracked allocation" with
   a nearby neighbour is an overrun (the gap tells you by how much).
2. Re-run with `SVOD_DEBUG_DISPATCH` set to see the exact VAs of the dispatch
   that faulted, and `RUST_LOG=svod_device=debug` to see the alloc/free history
   leading up to it.
3. Cross-reference the faulting VA against the dumped scratch/kernarg/buffer VAs.

The prime suspect for `NotPresent` is **scratch** (per the `Scratch` tag) — the
only shared, GPU-only, dynamically realloc'd-and-freed region, where a
realloc-vs-dispatch race can leave a kernel pointed at a freed buffer.

---

## Why this matters

Before the registry, a fault gave you a hex address and nothing else. Now the
fault *message itself* says whether the VA is live scratch, a freed/stale VA, or
wild — turning a blind hunt into a directed one. Paired with the poison latch
(which stops the device cleanly rather than letting corrupted state propagate)
and the dispatch dumps, the backend can localize a memory fault without
attaching a debugger to the GPU.
