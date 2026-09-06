---
sidebar_label: Profiling
---

# Profiling on CUDA

The [layered profiler](../../tile-kernels/profiling.md) is backend-neutral above
the `DispatchTimestamps` and `KernelResources` handles. This page is what the
CUDA backend puts into those handles, and which tiers exist.

| Tier | On CUDA | Source |
|---|---|---|
| **1 — device time** | yes | CUDA event pairs around each launch |
| **2 — roofline** | yes | backend-neutral (IR FLOP estimate, plan buffers) |
| **3 — static occupancy** | yes | `cuFuncGetAttribute` + `cuOccupancyMaxActiveBlocksPerMultiprocessor` |
| **4 — hardware counters** | **no** | needs CUPTI; not bound |

```bash
SVOD_DEVICE=CUDA:0 SVOD_PROFILE_ITERS=20 cargo run --release -p svod-model --example gigaam_infer -- ./audio.wav
```

---

## Tier 1: event timestamps

`CudaPlanCtx::dispatch` with `profile` set records a **timing event** before
and after the launch on the plan's stream and returns a
`CudaDispatchTimestamps` owning both. `timestamps_ns` must report nanoseconds
on the GPU clock, so it computes

```text
start    = cuEventElapsedTime(base_event, start_event)   // ms since the device opened
duration = cuEventElapsedTime(start_event, end_event)
end      = start + duration
```

The base event is recorded once at `CudaDevice::open` and is the zero of the
timeline. The duration is measured directly between the pair (full event
resolution, about half a microsecond); the absolute position goes through an
`f32` millisecond count that coarsens as the process ages, which is why `end`
is derived from `start` rather than measured against the base as well. Both
events must have completed (`cuEventQuery`) or the handle reports `None`.

Graph replays are profiled the same way: `replay_profiled` runs a chain
executable with an event-record node before and after every kernel and
returns one handle per captured kernel ([Architecture](./architecture.md)).

`Program::execute_timed`, used by BEAM, is the same event pair on the
dispatch stream, returned as a `Duration`.

---

## Tier 3: static resources

`CudaProgram::resource_usage` fills `KernelResources` from the function
attributes read at load:

| Column | Field | Source |
|---|---|---|
| `VGPR` | `vgprs` | `CU_FUNC_ATTRIBUTE_NUM_REGS` (registers per thread) |
| `SGPR` | `sgprs` | `-` (no scalar register file on NVIDIA) |
| `LDS` | `lds_bytes` | `CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES` (static `.shared`) |
| `scratch` | `scratch_bytes` | `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` (`.local` per thread) |
| `occ%` | `occupancy` | `cuOccupancyMaxActiveBlocksPerMultiprocessor(block) × block / max threads per SM` |

`wave_size` is the device's warp size (32). The occupancy query needs a block
size: the program remembers the block of its **latest launch** and falls back
to the function's `maxThreadsPerBlock` before any launch. Unlike the AMD
figure, which is register-limited only, the driver's count already folds in
registers, shared memory and the per-SM block limit.

---

## Tier 4: not available

There is no CUPTI binding, so `PlanContext::pmc_available()` is `false` on
CUDA. Setting `SVOD_PMC=1` does not fail: the profiler degrades to Tiers 1-3
and prints its one-line note that counters are unavailable. The `PmcCounter`
enum is AMD-SQ-specific today; widening it is part of the
[roadmap](./limitations.md).

For in-kernel timing experiments, `svod_codegen::llvm::nvptx::globaltimer()`
builds a `CUSTOM` node reading `%globaltimer`, the nanosecond GPU clock.
