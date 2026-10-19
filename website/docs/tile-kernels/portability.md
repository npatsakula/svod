---
sidebar_label: Portability
---

# Portability

## Targets

`atoms::Target` is everything the lowering knows about a GPU:

| Field | Meaning |
|---|---|
| `arch` | `GpuArch::{Cuda, Amd, Metal}` |
| `wave` | Lanes per warp or wave |
| `mma` | Matrix-core atoms with their operand layouts |
| `group_mma` | Warpgroup matrix atoms over shared-memory descriptors (CUDA sm_90) |
| `cp_async` | Asynchronous global → shared copies (CUDA sm_80+) |
| `ldmatrix` | Warp-collective 8×8 b16 fragment loads (CUDA sm_75+) |
| `smem_bytes` | Shared memory per block (the opt-in limit when the device reports it) |
| `sms` | SM or CU count, when the device reports it |

`Target::for_device(&spec)` resolves a target from a live device. `Target::for_arch(arch)`
builds one from an architecture alone, which is what host tests use. `atoms::sm86()` is the
RTX 3060 target with 28 SMs.

A target declares what the chip can do, never what it costs. What its primitives cost
(`atoms::Costs`: a shared-memory byte, a streamed byte, a scattered element, a matrix-core
step, a score's softmax work, a trip, a lane move, a shared round trip, a register copy, the
global latency) is measured on the device at first use (`probe::costs`) and kept in the tune
store; host tests declare it. There are no per-family cost tables. The costs reach the
lowering as `Lowering::costs` (orientation, exchange or round trip) and the op layer as
`Planner::device` (the candidate rankings).

| Target | Atoms | Fills | Tables | State |
|---|---|---|---|---|
| CUDA sm_80+ (sm_86) | `mma.sync` m16n8k16, `ldmatrix` | `cp.async`, 2–3 stages | yes | Measured on an RTX 3060 |
| CUDA sm_90 (Hopper) | `mma.sync` m16n8k16, `ldmatrix`; `wgmma` m64nNk16 over shared descriptors (GEMM) | `cp.async`, up to 227 KB shared | yes | Measured on an H100 PCIe: GEMM 4096³ bf16 433 TFLOP/s on the warpgroup core (234 on `mma.sync`; cuBLASLt 524) |
| AMD RDNA4 (gfx1200, gfx1201) | WMMA 16×16×16, 8 values per lane | register-staged, 1–2 stages | yes | Measured on an RX 9070 XT (gfx1201): GEMM 4096³ bf16 148 TFLOP/s (rocBLAS 80), attention d64 117 |
| AMD RDNA3 / RDNA3.5 (gfx1100–1102, gfx1151) | WMMA 16×16×16, replicated inputs | register-staged, 1–2 stages | yes | Measured on a Strix Halo (gfx1151): GEMM 4096³ bf16 28 TFLOP/s (rocBLAS 32.5), attention d64 28; gfx1100–1102 compiled, not run |
| AMD CDNA3 / CDNA4 (gfx942, gfx950) | MFMA 16×16×16, wave64 | register-staged, 1–2 stages, two crews | yes | Measured on an MI300X (gfx942): GEMM 4096³ bf16 555 TFLOP/s with two crews over a one-slot ring (hipBLASLt 615), attention d64 282, d128 343 at 32 query rows a wave; gfx950 compiled, not run |
| Apple | simdgroup 8×8×8 | none | no | Atoms only |

`ops::supported(device)` is true where config tables exist; on any other device each op builds
its graph fallback (`Fallback::Target`). Host tests check every atom's operand layouts against
the lane formulas of the vendor ISA documents, and for each target with tables they lower every
kernel family, interpret it before and after lowering, and compile it with `ptxas` or clang
when those are installed. "Compiled" is not "correct on the device": a target is measured only
once `targets::families_match_the_interpreter_on_the_device` passes on its hardware.

Without `cp.async` (RDNA has no global → LDS copy) a pipeline loads each step into registers,
computes the previous step, then writes the registers to shared memory, over two slots. The
same path runs on CUDA in a device test (`register_staged_families_match_on_cuda`).

## Strategy

The decided approach is **one shared tile program per op**. A kernel such as `kernels/gemm.rs`
is written once against tile values and never branches on the vendor. What varies per target
is data the lowering and the op layer choose:

| Per target | Where |
|---|---|
| Matrix-core atoms and their layouts | `atoms`, `layout/atoms.rs` |
| Copy mechanisms (`cp.async`, `ldmatrix`, register staging) | `Target` flags, the emitter |
| Schedule templates | `schedule::Schedule` |
| Config candidate rankings | `ops::config`, from the measured `Costs` |

A forked kernel program for one vendor is allowed only as a bounded exception.

## Not done

| Item | State |
|---|---|
| Hopper (sm_90a): TMA, mbarrier, warp-specialized template; `wgmma` for attention and its overlap across trips | Not started; the GEMM runs on `wgmma` with each product's chain waited for inside its statement, attention on `mma.sync` |
| Blackwell (sm_100a): tcgen05, tensor memory | Not started; no B200 access |
| AMD CDNA: MFMA 32×32, `buffer_load … lds` | Not started |
| AMD measurements | gfx1201 (RX 9070 XT) measured: GEMM, convolution and attention ahead of tk1 on Qwen3 and YOLO; gfx1151 (Strix Halo) measured: attention at d = 64 and 128; CDNA compiled, not run |
| Apple | Atoms only; no tables, no emitted kernels |
| Convolution (implicit GEMM) | Measured on sm_86, where YOLO26 runs on it channels-last (`ops::conv2d`); compiled only for AMD |
| Raw asm statements | Recorded in the IR, rejected by the lowering |
| fp8/int8 weights, GEMV + argmax, persistent grid, attention backward | Not started |
