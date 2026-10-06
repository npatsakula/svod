---
sidebar_label: Architecture
---

# Architecture

This page follows the backend from the driver binding up to graph replay.
Everything below is in `device/src/cuda/` unless noted.

```text
mod.rs        the dl_api! runtime-binding macro and the module's re-exports
sys.rs        the bound driver entry points (libloading), CUresult, handles
device.rs     CudaDevice: primary context, limits, lanes, base event, scoped-sync tables, poison latch
allocator.rs  CudaAllocator: device / managed / pinned memory, staged copies
program.rs    CudaProgram: cubin or PTX module load, cuLaunchKernel, execute_timed, resources
sync.rs       CudaPlanCtx, CudaDispatchTimestamps, CudaCompletionToken
graph.rs      CudaGraph: a CUDA graph DAG from GraphKernel::deps, patched replays
cupti.rs      the CUPTI range profiler behind Tier 4 (see Profiling)
```

---

## Driver bindings

`sys.rs` declares every entry point the backend uses in one `dl_api!` block —
the Rust field, the exact export name, and the C prototype; the macro itself
lives in `mod.rs` and `cupti.rs` reuses it. `Api::load` opens `libcuda.so.1`
and resolves all of them up front, so a missing or renamed symbol fails once,
at load, as `Error::DeviceUnavailable` (`libcuda.so.1 has no symbol ...`)
rather than at first use. Names are the **versioned** exports that `cuda.h`
remaps to: `cuMemAlloc_v2`, `cuDevicePrimaryCtxRelease_v2`,
`cuGraphAddKernelNode_v2`, `cuGraphExecKernelNodeSetParams_v2`,
`cuGraphInstantiateWithFlags` (the unversioned `cuGraphInstantiate` is a legacy
five-argument ABI and is never touched). `cuEventElapsedTime` is the deliberate
exception: its `_v2` is CUDA 12.8, which would raise the driver floor to R570.

Handles are `#[repr(transparent)]` pointer newtypes (`CUcontext`, `CUmodule`,
`CUfunction`, `CUstream`, `CUevent`, `CUgraph`, `CUgraphExec`, ...);
`CUdeviceptr` is `u64`. `CUresult` is an integer newtype so codes from a newer
driver still round-trip; `CUresult::check("cuLaunchKernel")` turns a failure
into

```text
CUDA cuLaunchKernel failed: CUDA_ERROR_INVALID_VALUE (1): invalid argument
```

using the driver's own `cuGetErrorName` / `cuGetErrorString`. The
`CudaKernelNodeParams` struct mirrors `CUDA_KERNEL_NODE_PARAMS_v2` with
compile-time size and offset assertions.

---

## Device, context, streams

`CudaDevice::open(id)` is cached per process. It runs `cuInit`, retains the
device's **primary context** (`cuDevicePrimaryCtxRetain`), reads the
`CudaLimits` it needs (`cuDeviceGetAttribute`: SM count, threads per block and
per SM, blocks per SM, shared memory per block, warp size, and whether managed
memory is both supported and concurrently accessible), creates two non-blocking
streams (a **copy stream** for the allocator and a **dispatch stream** for
per-call `Program::execute`), and records one **base event** on the legacy
default stream — waited before the device is returned — that is the zero of
every GPU-clock timestamp.

The driver keeps the current context per thread, so every entry point of the
backend starts with `enter()`: refuse if the device is poisoned, then
`cuCtxSetCurrent`. A **sticky** `CUresult` (`ILLEGAL_ADDRESS`,
`LAUNCH_FAILED`, `ILLEGAL_INSTRUCTION`, `MISALIGNED_ADDRESS`,
`ECC_UNCORRECTABLE`, `LAUNCH_TIMEOUT`, `ASSERT`, `HARDWARE_STACK_ERROR`,
`INVALID_ADDRESS_SPACE`, `INVALID_PC`, `DEINITIALIZED`, and conservatively
`UNKNOWN` — `sys.rs`) latches the poison flag with its message; every later
call on the device fails fast with that message, as on AMD.

---

## Memory

A `RawBuffer::Cuda` carries a device pointer, an optional host pointer, and its
`CudaMemory` kind, chosen from the `BufferSpec`:

| `BufferSpec` | Kind | Driver call |
|---|---|---|
| default | `Device` | `cuMemAlloc` — device memory, no host mapping |
| `cpu_access` | `Managed` when the device reports concurrent managed access, else `Pinned` | `cuMemAllocManaged(ATTACH_GLOBAL)`, one address valid on both sides; else `cuMemHostAlloc(PORTABLE \| DEVICEMAP)` + `cuMemHostGetDevicePointer` |
| `host` | `Pinned` | `cuMemHostAlloc(PORTABLE \| DEVICEMAP)`, kernels read it over the bus |

`supports_device_local()` is `true`, so intermediates stay on the device.
Host <-> device copies first wait the storage's in-flight producers and
readers (`CudaDevice::wait_storage`, below — host access is not ordered
against the lanes), then move data. Up to 4 MiB a copy-out is one synchronous
`cuMemcpyDtoH`, while a copy-in is a `cuMemcpyHtoDAsync` on the copy lane
published as the storage's new producer; it synchronizes the stream only when
the source is memory the driver tracks (pinned, registered or managed, asked
with `cuPointerGetAttribute`), because the driver stages a pageable source
before it returns. Above 4 MiB both directions go in 4 MiB chunks through a
lazily allocated **pinned staging buffer** with `cuMemcpyHtoDAsync` /
`cuMemcpyDtoHAsync`, synchronizing the stream per chunk. Pinned buffers are
`memcpy`'d directly. Device-to-device `_transfer` and zero-fills are asynchronous on
the copy lane: ordered after the producers with `cuStreamWaitEvent`,
published as the new producer of both ranges, and waited by every later
launch on any lane, so they do not block the host — except an overlapping
range inside one allocation, which bounces through a temporary to keep
`memmove` semantics and waits when that temporary is freed. Freeing waits the
storage's producers first; if the wait fails
(poisoned context) the allocation is **quarantined** (leaked) rather than
freed under an in-flight kernel. Like every compute allocator it sits under
`LruAllocator`, which fences a recycled allocation on its previous owner's
producers.

---

## Programs and launches

`CudaProgram::load` branches on `is_cubin` — an ELF image goes through
`validate_cubin`, PTX text through the entry's `.param` check — and both reach
the same `cuModuleLoadDataEx` with 16 KiB
error and info log buffers, so a JIT failure surfaces as
`Error::CudaJit { kernel, cause, log }` carrying `ptxas`'s own message (see
[Debugging](./debugging.md)); the info log goes to `tracing::debug!`. It then
binds the entry with `cuModuleGetFunction` and reads the function attributes
`MAX_THREADS_PER_BLOCK`, `NUM_REGS`, `SHARED_SIZE_BYTES`, `LOCAL_SIZE_BYTES`.
The module is `Arc`-shared with any graph that captured it and unloaded on the
last drop.

Kernel arguments travel as **one packed blob** in `cuLaunchKernel`'s `extra`
array (`CU_LAUNCH_PARAM_BUFFER_POINTER` / `_SIZE` / `_END`), laid out by the
shared `ClikeKernargLayout`: 8-byte device pointers, 4-byte `i32` scalars, in
PARAM slot order, which is exactly PTX's natural `.param` layout. `global_size`
is the **grid in blocks** and `local_size` the **block in threads** (the
work-group convention AMD and Metal use); a block larger than the function's
`maxThreadsPerBlock` is rejected before launch with the register, shared and
local memory figures in the message.

`Program::execute` launches on the device's dispatch stream and optionally
waits on it; `execute_timed` records a timing event pair around the launch and
returns `cuEventElapsedTime`, so BEAM ranks candidates on GPU time.

---

## Plan contexts, tokens, timelines

Each execution plan gets a `CudaPlanCtx`: **one non-blocking stream**, which
is its lane, plus the CUPTI counter selection and session when counters are
armed. `dispatch` launches on it; with `profile` set it brackets the
launch with timing events and returns a `CudaDispatchTimestamps`
([Profiling](./profiling.md)). `completion_token` records a completion-only
event (`CU_EVENT_DISABLE_TIMING`) whose `wait` is `cuEventSynchronize` and
whose `retired` is `cuEventQuery`; `synchronize` is `cuStreamSynchronize`.

### Scoped synchronization

Lanes are not ordered against each other, so `CudaDevice` keeps three
tables (module docs of `device/src/cuda/device.rs`):

- **producers** — storage base -> the newest completion token per lane that
  read or wrote it (a host overwrite is a WAR hazard against in-flight
  readers too). The execution plan publishes its or its graph's token on every
  storage it touches after each execute; the allocator publishes a
  copy-lane token after each transfer or memset. `wait_storage(base)` drains
  the lanes below, then waits those tokens, then drops them from the table. A
  storage the table does not know — including one whose newest token belongs
  to another backend — falls back to `cuCtxSynchronize`.
- **lanes** — every live lane and how many submissions it holds that no token
  has been published for (per-call `Program::execute`, a plan that failed
  mid-way, a graph replay before its token is fetched). `wait_storage` drains
  such lanes on the host; a copy instead records a tail event on each and
  waits it on the GPU. The copy lane itself is not in this table.
- **copy tail** — the newest copy-lane event; each launch waits it on the
  GPU before running, so asynchronous copies precede every later kernel.

`SVOD_CUDA_SCOPED_SYNC=0` disables all of it: every wait drains the context
and every copy synchronizes the copy stream.

The executor's cross-plan ordering is a host signal (`CpuTimelineSignal`) on
every backend, CUDA included; there is no `TimelineSignal` implementation of
its own. What keeps the host signal off the critical path is the machinery
above: the tables order GPU work against GPU work with `cuStreamWaitEvent`, so
a plan only ever waits on the host for what it actually reads.

---

## Graphs

`CudaGraph::capture` turns a captured kernel chain into a real **CUDA graph**:
one `cuGraphAddKernelNode_v2` per kernel whose dependency list is exactly
`GraphKernel::deps`, the host hazard analysis. Independent kernels may
therefore overlap on the device (the AMD backend discards `deps` because a
single in-order ring makes them redundant). Each node's params point at that
kernel's kernarg blob through the same `extra` protocol as eager launches; the
graph is instantiated with `cuGraphInstantiateWithFlags`. Capture declines
(`Ok(None)`) for an empty chain, a non-CUDA program, or a program of another
device, and fails (`Err`) on a poisoned device or a dependency on a later
kernel; the plan treats both as per-call dispatch.

`replay(buffers, vals)` re-packs only the kernels whose `(buffers, vals)` slice
changed and updates those nodes with `cuGraphExecKernelNodeSetParams_v2`,
then `cuGraphLaunch`es on the graph's own stream. One subtlety: the recorded
hazards are only valid for the **aliasing** the chain was captured with. If a
replay binds buffers so that a different pair of slots now shares an address,
the graph switches to a lazily built **capture-order chain** (each kernel
after the previous one), which is always correct.

`replay_profiled` uses a third executable, the chain with an
`cuGraphAddEventRecordNode` pair around every kernel; the events are re-armed
per launch (`cuGraphExecEventRecordNodeSetEvent`) so handles already handed
out keep their stamps, and one `CudaDispatchTimestamps` per captured kernel
is returned in capture order.

---

## Object cache identity

Compiled PTX goes through the shared on-disk object cache, keyed by the
rendered IR and a `CompilerIdentity`:

```text
schema:              OBJECT_CACHE_SCHEMA
backend:             nvptx-clang
target_architecture: nvptx64-nvidia-cuda/sm_86
toolchain:           <clang identity>[;ptxas:path=...;version=...]
flags:               -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 --cuda-feature=+ptx78 -Wno-override-module - -o -
                     [-arch=sm_86 -o /dev/stdout /dev/stdin]
abi:                 ptx-kernel-abi-v1;warp-size=32
object_format:       ptx-text-v1 | cubin-v1
```

The bracketed halves are the `ptxas` path: with the assembler the cached
object is a **cubin**, without it **PTX text** the driver assembles at load,
keeping the SASS in its own `~/.nv/ComputeCache`. The two formats never share
an entry, since they differ in `toolchain`, `flags` and `object_format` alike.
The rendered IR is not the whole key either: the ABI descriptors are appended
to it, because a cubin's entry is checked against them at compile time. Every
cache hit is re-validated by its format's validator — `validate_cubin` or
`validate_ptx`, see [Codegen](./codegen.md) — before it reaches the driver.
`SVOD_OBJECT_CACHE=0` disables the cache, `SVOD_OBJECT_CACHE_DIR` relocates
it and `SVOD_OBJECT_CACHE_MAX_BYTES` sets its budget (see the
[CPU page](../cpu.md)).

The device factory (`create_cuda_device`) also refuses a device whose
per-block shared memory limit is below the optimizer profile's static
`shared_max`, since a kernel sized against the profile would otherwise only
fail at JIT.
