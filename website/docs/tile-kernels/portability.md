---
sidebar_label: Portability
---

# Portability

## Targets

`atoms::Target` holds what a GPU can do: its matrix-core atoms, how global memory reaches a
shared ring (`Fill::CpAsync` or `Fill::Staged`), `ldmatrix` and transposing loads, lane moves,
which syncs are instructions (`SyncCaps`), occupancy, shared memory and SM count; see its
rustdoc. `Target::for_device(&spec)` resolves a target from a live device, `Target::for_arch`
from an architecture alone (what host tests use), and `Target::offline(arch, sms, xcds)` or
`Target::card(name)` describes a card this machine has no device for.

A target declares what the chip can do, never what it costs. What its primitives cost
(`atoms::Costs`) is measured on the device at first use (`probe::costs`) and kept in the tune
store; host tests declare it ([Tuning](./tuning#device-costs)). There are no per-family cost
tables. The costs reach the lowering as `Lowering::device` (orientation, exchange or round trip)
and the op layer as the `Planner`'s device (the candidate rankings). The measured targets and
their numbers are in the [Overview](./overview#targets).

`ops::supported(device)` is true when `device` is the default device, its target has tk
kernels (`config::has_kernels`) and reports an SM count; the cost probes run on the first ask.
On any other device each op builds its graph fallback (`Fallback::Target`). Host tests check
every atom's operand layouts against the lane formulas of the vendor ISA documents, and for each
target with kernels they lower every kernel family, interpret it before and after lowering, and
compile it with `ptxas` or clang when those are installed. "Compiled" is not "correct on the
device": a target is measured only once `targets::families_match_the_interpreter_on_the_device`
passes on its hardware.

Without `cp.async` (AMD has no asynchronous global → LDS copy) a pipeline loads each step into
registers, computes the previous step, then writes the registers to shared memory, over one or
two slots. The same path runs on CUDA in a device test (`register_staged_families_match_on_cuda`).

## Strategy

The decided approach is **one shared tile program per op**. A kernel such as `kernels/gemm.rs`
is written once against tile values and never branches on the vendor. What varies per target
is data the lowering and the op layer choose:

| Per target | Where |
|---|---|
| Matrix-core atoms and their layouts | `atoms`, `layout/atoms.rs` |
| Copy mechanisms (`cp.async`, `ldmatrix`, register staging, transposing loads) | `Target` fields, the emitter |
| Schedule templates | `schedule::Schedule` |
| Config candidate rankings | `ops::config`, from the measured `Costs` |

A forked kernel program for one vendor is allowed only as a bounded exception.

## Not in this release

| Item | State |
|---|---|
| Hopper TMA, mbarrier, warp-specialized producers | Not started; GEMM and attention run `wgmma` fed by `cp.async` |
| Blackwell `tcgen05`, tensor memory | Not started; sm_120 runs the sm_80 path |
| CDNA MFMA 32×32, `buffer_load … lds` | Not started (32×32 layouts exist, no atom) |
| Two-crew ping-pong off CDNA | `Target::crews` is 2 only where the matrix core runs beside the vector units and a lane's file holds two crews' accumulators (CDNA) |
| Apple | Atoms only; no kernels |
| Convolution image-staged form, depthwise, grouped | Graph |
| fp8/int8, attention backward | Not started |
