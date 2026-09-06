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
| **Hardware counters (Tier 4)** | No CUPTI; `pmc_available()` is `false`, `SVOD_PMC` degrades with a note. | [Profiling](./profiling.md) |
| **Tile kernels (`tk`)** | AMD-only: `resolve_arch` yields no `AmdArch` for a CUDA spec, so a `tk` launch reports `UnsupportedArch`. | `tk/src/target.rs` |
| **Dynamic shared memory** | Launches pass `shared_mem_bytes = 0`; only static `.shared` is used and `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` is never called, so a kernel needing more than the default per-block limit fails at JIT. The device factory refuses a device whose limit is below the profile's `shared_max` (48 KiB) up front. | `device/src/cuda/program.rs`, `runtime/src/devices/cuda.rs` |
| **Hopper / Blackwell matrix paths** | Only `mma.sync` (`m16n8kK`) is lowered; no `wgmma`, no `tcgen05`. | `codegen/src/llvm/nvptx/wmma.rs` |
| **Pre-assembled objects** | The object cache stores PTX text; every fresh load pays the driver JIT (cached by the driver in `~/.nv/ComputeCache`). A `ptxas` pre-assembly (`object_format: cubin`) is not wired. | `runtime/src/devices/cuda.rs` |
| **Userspace NV driver** | Tinygrad's `ops_nv` (direct GPU-FIFO submission) needs a generated ABI per driver branch; Svod stays on the stable `libcuda.so.1` API. `NV` is accepted as an alias of `CUDA` in `SVOD_DEVICE` and reserved for that future backend. | `nvidia_backend_plan.md` |

Numerical notes rather than gaps: f64 `Exp2` / `Log2` and all transcendentals
take the polynomial path ([Codegen](./codegen.md)); `lg2.approx.f32` is
available to the renderer but not used by ordinary graphs.

---

## Requirements that are not negotiable today

- The driver must be at least CUDA 12.0 / R525: the CUDA-graph entry points
  are bound by their 12.0 versioned names. The PTX ISA is pinned to 7.8 by
  `--cuda-feature=+ptx78`, so a newer clang does not raise that floor.
- `clang` must carry the NVPTX target; there is no NVRTC fallback.

---

## Roadmap

In the order the plan (`nvidia_backend_plan.md`, phase 5) lists them:

1. **Stream-ordered frees**: `cuMemFreeAsync` on the copy lane for device
   memory, so a free stops draining the device.
2. **Real P2P**: bind `cuDeviceCanAccessPeer` / `cuCtxEnablePeerAccess` /
   `cuMemcpyPeerAsync` and route `SyncStrategy::PeerToPeer` through them.
3. **CUPTI counters**: widen `PmcCounter` beyond the AMD SQ set and add a
   Tier 4 provider.
4. **fp8**: lower the sm_89 `cvt` intrinsics so the fp8 `mma.sync` rows become
   reachable.
5. **`ptxas` pre-assembly** when the toolkit is present, cached as a cubin.
6. **Dynamic shared memory** via `cuFuncSetAttribute`, and `tk` over
   `GpuArch` so tile kernels run on CUDA.
