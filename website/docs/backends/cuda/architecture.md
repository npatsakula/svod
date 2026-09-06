---
sidebar_label: Architecture
---

# Architecture

This page follows the backend from the driver binding up to graph replay.
Everything below is in `device/src/cuda/` unless noted.

```text
sys.rs        the bound driver entry points (libloading), CUresult, handles
device.rs     CudaDevice: primary context, limits, streams, base event, poison latch
allocator.rs  CudaAllocator: device / managed / pinned memory, staged copies
program.rs    CudaProgram: PTX module load, cuLaunchKernel, execute_timed, resources
sync.rs       CudaPlanCtx, CudaDispatchTimestamps, CudaCompletionToken, CudaTimelineSignal
graph.rs      CudaGraph: a CUDA graph DAG from GraphKernel::deps, patched replays
```

---

## Driver bindings

`sys.rs` declares every entry point the backend uses in one `cuda_api!`
macro: the Rust field, the exact export name, and the C prototype. `Api::load`
opens `libcuda.so.1` and resolves all of them up front, so a missing or renamed
symbol fails once, at load, as `Error::DeviceUnavailable` (`libcuda.so.1 has no
symbol ...`) rather than at first use. Names are the **versioned** exports that
`cuda.h` remaps to: `cuMemAlloc_v2`, `cuCtxDestroy_v2`,
`cuGraphAddKernelNode_v2`, `cuGraphExecKernelNodeSetParams_v2`,
`cuGraphInstantiateWithFlags` (the unversioned `cuGraphInstantiate` is a legacy
five-argument ABI and is never touched).

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
`CudaLimits` it needs (`cuDeviceGetAttribute`: SM count, threads and shared
memory per block and per SM, registers, warp size, L2, managed-memory
support), creates two non-blocking streams (a **copy stream** for the
allocator and a **dispatch stream** for per-call `Program::execute`), and
records one **base event** that is the zero of every GPU-clock timestamp.

The driver keeps the current context per thread, so every entry point of the
backend starts with `enter()`: refuse if the device is poisoned, then
`cuCtxSetCurrent`. A **sticky** `CUresult` (`ILLEGAL_ADDRESS`,
`LAUNCH_FAILED`, `ILLEGAL_INSTRUCTION`, `ECC_UNCORRECTABLE`, ... the codes the
driver documents as fatal to the context) latches the poison flag with its
message; every later call on the device fails fast with that message, as on
AMD.

---

## Memory

A `RawBuffer::Cuda` carries a device pointer, an optional host pointer, and its
`CudaMemory` kind, chosen from the `BufferSpec`:

| `BufferSpec` | Kind | Driver call |
|---|---|---|
| default | `Device` | `cuMemAlloc` — device memory, no host mapping |
| `cpu_access` | `Managed` when the device reports concurrent managed access, else `Pinned` (WDDM, pre-Pascal) | `cuMemAllocManaged`, one address valid on both sides |
| `host` | `Pinned` | `cuMemHostAlloc(PORTABLE \| DEVICEMAP)`, kernels read it over the bus |

`supports_device_local()` is `true`, so intermediates stay on the device.
Host <-> device copies first wait the storage's in-flight producers and
readers (`CudaDevice::wait_storage`, below — host access is not ordered
against the lanes), then move data with one synchronous `cuMemcpy` up to
4 MiB and above that in 4 MiB chunks through a lazily allocated **pinned
staging buffer** with `cuMemcpyHtoDAsync` / `cuMemcpyDtoHAsync` on the copy
stream, synchronizing the stream per chunk. Pinned buffers are `memcpy`'d
directly. Device-to-device `_transfer` and zero-fills are asynchronous on
the copy lane: ordered after the producers with `cuStreamWaitEvent`,
published as the new producer of both ranges, and waited by every later
launch on any lane, so they never block the host; an overlapping range
inside one allocation bounces through a temporary to keep `memmove`
semantics. Freeing waits the storage's producers first; if the wait fails
(poisoned context) the allocation is **quarantined** (leaked) rather than
freed under an in-flight kernel. Like every compute allocator it sits under
`LruAllocator`, which fences a recycled allocation on its previous owner's
producers.

---

## Programs and launches

`CudaProgram::load` hands the PTX text to `cuModuleLoadDataEx` with 16 KiB
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
`maxThreadsPerBlock` is rejected before launch with the register and shared
memory figures in the message.

`Program::execute` launches on the device's dispatch stream and optionally
waits on it; `execute_timed` records a timing event pair around the launch and
returns `cuEventElapsedTime`, so BEAM ranks candidates on GPU time.

---

## Plan contexts, tokens, timelines

Each execution plan gets a `CudaPlanCtx`: **one non-blocking stream**, which
is its lane. `dispatch` launches on it; with `profile` set it brackets the
launch with timing events and returns a `CudaDispatchTimestamps`
([Profiling](./profiling.md)). `completion_token` records a completion-only
event (`CU_EVENT_DISABLE_TIMING`) whose `wait` is `cuEventSynchronize` and
whose `retired` is `cuEventQuery`; `synchronize` is `cuStreamSynchronize`.

### Scoped synchronization

Lanes are not ordered against each other, so `CudaDevice` keeps three
tables (module docs of `device/src/cuda/device.rs`):

- **producers** — storage base -> the newest completion token per lane that
  read or wrote it (a host overwrite is a WAR hazard against in-flight
  readers too). The executor publishes a plan's or graph's token on every
  storage the plan touches after each execute; the allocator publishes a
  copy-lane token after each transfer or memset. `wait_storage(base)` waits
  those tokens only. A storage the table does not know falls back to
  `cuCtxSynchronize`.
- **lanes** — every live lane and whether it holds submissions no token
  has been published for (per-call `Program::execute`, a plan that failed
  mid-way, a graph replay before its token is fetched); every scoped wait
  drains such lanes.
- **copy tail** — the newest copy-lane event; each launch waits it on the
  GPU before running, so asynchronous copies precede every later kernel.

`SVOD_CUDA_SCOPED_SYNC=0` disables all of it: every wait drains the context
and every copy synchronizes the copy stream.

The executor's cross-plan ordering uses `CudaTimelineSignal`, a timeline
published by events: `signal(stream, value)` records an event at the stream's
tail and files `(value, event)`; `value()` folds every retired publication
into the floor; `wait(target)` blocks on the event carrying the smallest value
`>= target`; `wait_on_stream` orders GPU work after it with
`cuStreamWaitEvent`. Slots are `Arc`'d so a waiter keeps its event alive while
another thread folds it away.

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
device.

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
backend:             nvptx-clang
target_architecture: nvptx64-nvidia-cuda/sm_86
toolchain:           <clang identity>
flags:               -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 -Wno-override-module - -o -
abi:                 ptx-kernel-abi-v1;warp-size=32
object_format:       ptx-text-v1
```

The cache stores **PTX text**, never a cubin: the driver assembles it at load
and keeps the SASS in its own `~/.nv/ComputeCache`. Every cache hit is
re-validated (`validate_ptx`) before it reaches the driver, see
[Codegen](./codegen.md). `SVOD_OBJECT_CACHE=0` disables the cache and
`SVOD_OBJECT_CACHE_DIR` relocates it.

The device factory (`create_cuda_device`) also refuses a device whose
per-block shared memory limit is below the optimizer profile's static
`shared_max`, since a kernel sized against the profile would otherwise only
fail at JIT.
