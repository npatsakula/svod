---
sidebar_label: Profiling
---

# Profiling on CUDA

The layered profiler (`ExecutionPlan::profile`) is backend-neutral above
the `DispatchTimestamps` and `KernelResources` handles. This page is what the
CUDA backend puts into those handles, and which tiers exist.

| Tier | On CUDA | Source |
|---|---|---|
| **1 — device time** | yes | CUDA event pairs around each launch |
| **2 — roofline** | yes | backend-neutral (IR FLOP estimate, plan buffers) |
| **3 — static occupancy** | yes | `cuFuncGetAttribute` + `cuOccupancyMaxActiveBlocksPerMultiprocessor` |
| **4 — hardware counters** | yes | CUPTI range profiler (`libcupti.so.13`) |

```bash
SVOD_DEVICE=CUDA:0 cargo run --release -p svod-model --example gigaam_infer -- --profile ./audio.wav
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
| `occ%` | `occupancy` | `cuOccupancyMaxActiveBlocksPerMultiprocessor(block) × block / max threads per SM`, clamped to 1 |

`wave_size` is the device's warp size (32). The occupancy query needs a block
size: the program remembers the block of its **latest launch** and falls back
to the function's `maxThreadsPerBlock` before any launch. Unlike the AMD
figure, which is register-limited only, the driver's count already folds in
registers, shared memory and the per-SM block limit.

---

## Tier 4: hardware counters

Counters come from the CUPTI range profiler, bound at runtime from one
soname, `libcupti.so.13`, looked up on the loader path and then in
`/opt/cuda/lib64`, `/usr/local/cuda/extras/CUPTI/lib64` and
`$CUDA_PATH/{lib64,extras/CUPTI/lib64}` (`device/src/cuda/cupti.rs`). CUDA 13
folded the PerfWorks host API into CUPTI, so that one library carries the
whole sequence — CUPTI `dlopen`s `libnvperf_host.so` itself, which therefore
has to be resolvable by the loader, and a `CUPTI_ERROR_NOT_INITIALIZED` is
what it looks like when it is not. The binding is optional the way `ptxas` is:
when it is absent, unusable, or disabled with `SVOD_CUDA_CUPTI=0`,
`pmc_available()` is `false` and the profiler degrades to Tiers 1-3 with its
one-line note.

Two of the params structs grew in CUDA 13.3, so each call sends the newest
`struct_size` first and steps back one size on `CUPTI_ERROR_INVALID_PARAMETER`
— `cuptiProfilerGetCounterAvailability` (41 then 40) and
`cuptiProfilerHostInitialize` (56 then 48) — and `abi_ladder` remembers the
first size the installed CUPTI did not reject as a bad parameter.

`SVOD_PMC=1` selects the backend default:

| Token | Metric | Meaning |
|---|---|---|
| `cycles` | `sm__cycles_active.sum` | cycles with at least one warp resident |
| `warps` | `sm__warps_launched.sum` | warps launched |
| `inst` | `smsp__inst_executed.sum` | warp instructions executed |
| `tensor` | `sm__pipe_tensor_cycles_active.sum` | cycles the tensor pipe was active |
| `dram` | `dram__bytes.sum` | bytes moved through DRAM |

Name a subset by token — `SVOD_PMC=tensor,dram`. Tokens are unique across
backends, so an AMD token on CUDA is dropped rather than mis-programming a
block; `set_pmc` returns how many it kept and the executor says so on stderr:

```text
SVOD_PMC: 2 of 5 requested counters are not collected on this backend
```

A list whose tokens are all unknown falls back to the default set, and a
selection nothing survives means an ordinary un-armed timing run. `tensor`
against `cycles` is the tensor-core utilization of a matmul or a
flash-attention kernel; `dram` is what separates a bandwidth-bound kernel
from an issue-bound one.

```bash
SVOD_DEVICE=CUDA:0 SVOD_PMC=1 cargo run --release -p svod-model --example gigaam_infer -- --profile ./audio.wav
```

The tk3 throughput probes time the hand kernels alone. They read event timestamps, not
counters:

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk3 --release --lib gemm_throughput_probe -- --ignored --nocapture
```

### Privileges

The driver restricts counter collection to admin users by default, and the
restriction is not where it first appears: `cuptiRangeProfilerEnable` and
`cuptiRangeProfilerSetConfig` both succeed without it, and only the counter
availability image and `cuptiRangeProfilerStart` fail with
`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`. `pmc_available()` therefore probes the
availability image. To lift the restriction:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  | sudo tee /etc/modprobe.d/nvidia-profiling.conf
# rebuild the initramfs for your distro, then reboot
```

`scripts/cupti_probe.cu` runs the whole sequence standalone against a saxpy
kernel and reports where it stops, which is the quickest way to tell a
privilege problem from a toolkit one; running it under `sudo` confirms the
diagnosis without a reboot.

### What collection costs

Capture runs in `CUPTI_AutoRange` with `CUPTI_KernelReplay`: CUPTI opens one
range per launch and replays the kernel internally to cover a multi-pass
config (the five counters above have scheduled in one pass on every chip we
have run; a set that needs more is caught at `Stop` and degrades to timing).
Two consequences are handled for you:

- A captured CUDA graph replays as one opaque submission and its handles
  never carry counters, so a counted run takes the per-dispatch path.
- Kernel replay inflates the dispatch's own event pair by orders of magnitude,
  so a counted run adds one disarmed pass; `merge_min` keeps its timing next
  to the counted pass's counters. Timing and counters in one table therefore
  come from different passes.

Readback is host-driven and one session cannot overlap the next, so a counted
dispatch synchronizes in place. Any CUPTI failure degrades that dispatch to
timing only rather than failing the run.

For in-kernel timing experiments, `svod_codegen::llvm::nvptx::globaltimer()`
builds a `CUSTOM` node reading `%globaltimer`, the nanosecond GPU clock.
