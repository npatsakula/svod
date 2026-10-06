---
sidebar_label: Limitations & Roadmap
---

# Limitations and Roadmap

What the backend does not do yet, with the concrete reason in the source, and
what is planned. Nothing here fails silently: each gap is either a clean error
or a documented fallback.

---

## Not implemented

| Gap | Today | Where |
|---|---|---|
| **fp8 conversions** | A cast to or from `FP8E4M3` / `FP8E5M2` fails at render (`NVPTX fp8 cast ...`); the sm_89 `cvt.*.e4m3x2` intrinsics are not emitted. The fp8 `mma.sync` rows exist in `resolve_mma` but cannot be fed. | `codegen/src/llvm/nvptx/ops.rs` |
| **Stream-ordered frees** | `cuMemFree*` synchronizes the whole device and blocks every other thread's driver call meanwhile; `_free` is rare under `LruAllocator`, but `cuMemFreeAsync` is not bound. | `device/src/cuda/allocator.rs` |
| **Peer-to-peer copies** | `cuMemcpyPeerAsync` / `cuDeviceCanAccessPeer` are not bound. A `CUDA:0 → CUDA:1` copy takes `SyncStrategy::PeerToPeer` in the executor, which falls back to `Buffer::copy_from`; two allocators are two devices, so the bytes bounce through a host `Vec`. | `runtime/src/executor.rs`, `device/src/buffer.rs` |
| **Dynamic shared memory** | Launches pass `shared_mem_bytes = 0`; only static `.shared` is used and `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` is never called, so a kernel needing more than the default per-block limit fails at JIT. The device factory refuses a device whose limit is below the profile's `shared_max` (48 KiB) up front. | `device/src/cuda/program.rs`, `runtime/src/devices/cuda.rs` |
| **Hopper / Blackwell matrix paths** | Only `mma.sync` (`m16n8kK`) is lowered; no `wgmma`, no `tcgen05`. | `codegen/src/llvm/nvptx/wmma.rs` |
| **Cubins on hosts without `ptxas`** | With no CUDA toolkit the object cache stores PTX text and every fresh load pays the driver JIT (cached by the driver in `~/.nv/ComputeCache`). There is no bundled assembler: `ptxas` is used when it is installed (`object_format: cubin-v1`), otherwise the driver does the work. | `runtime/src/cuda/compile.rs` |
| **Userspace NV driver** | Tinygrad's `ops_nv` (direct GPU-FIFO submission) needs a generated ABI per driver branch; Svod stays on the stable `libcuda.so.1` API. `NV` is deliberately *not* accepted in `SVOD_DEVICE` (only `CUDA` and `GPU` are); the name is reserved for that future backend. | `device/src/registry.rs` |

Numerical notes rather than gaps: f64 `Exp2` / `Log2` and all transcendentals
take the polynomial path ([Codegen](./codegen.md)); `lg2.approx.f32` is
available to the renderer but not used by ordinary graphs.

---

## Requirements that are not negotiable today

- The driver must be at least CUDA 12.0 / R525: the CUDA-graph entry points
  are bound by their 12.0 versioned names, and an older driver leaves the
  backend silently unregistered. The PTX ISA pin (`--cuda-feature`)
  follows the compute capability — 7.8 up to sm_88, 8.4 on sm_89 and sm_90
  (CUDA 12.4 / R550), then 8.6, 8.7 and 8.8 across Blackwell (up to CUDA 12.9)
  — so a Blackwell part raises the floor to the driver of its own ISA, but a
  newer clang does not.
- `clang` must carry the NVPTX target; there is no NVRTC fallback.

---

## Roadmap

What is left, in priority order:

1. **Stream-ordered frees**: `cuMemFreeAsync` on the copy lane for device
   memory, so a free stops draining the device.
2. **Real P2P**: bind `cuDeviceCanAccessPeer` / `cuCtxEnablePeerAccess` /
   `cuMemcpyPeerAsync` and route `SyncStrategy::PeerToPeer` through them.
3. **fp8**: lower the sm_89 `cvt` intrinsics so the fp8 `mma.sync` rows become
   reachable, and let `for_cuda_arch` build the sm_89 profile.
4. **Dynamic shared memory** via `cuFuncSetAttribute`, so a kernel may exceed
   the default 48 KiB per-block limit.

Scoped synchronization and CUPTI hardware counters were the other two items;
both have shipped ([Architecture](./architecture.md),
[Profiling](./profiling.md)).
