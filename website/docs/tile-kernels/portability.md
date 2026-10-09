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
| `cp_async` | Asynchronous global → shared copies (CUDA sm_80+) |
| `ldmatrix` | Warp-collective 8×8 b16 fragment loads (CUDA sm_75+) |
| `smem_bytes` | Shared memory per block (the opt-in limit when the device reports it) |
| `sms` | SM or CU count, when the device reports it |

`Target::for_device(&spec)` resolves a target from a live device. `Target::for_arch(arch)`
builds one from an architecture alone, which is what host tests use. `atoms::sm86()` is the
RTX 3060 target with 28 SMs.

| Target | Atoms and layouts | Config tables | Lowered and run |
|---|---|---|---|
| CUDA sm_80+ (measured on sm_86) | `mma.sync` m16n8k16, `ldmatrix`, `cp.async` | yes | yes |
| AMD CDNA (gfx942) | MFMA 16×16×16 | no | no |
| AMD RDNA3 / RDNA4 | WMMA 16×16×16 | no | no |
| Apple | simdgroup 8×8×8 | no | no |

`ops::supported(device)` is true only where config tables exist, so on every other device each
op builds its graph fallback (`Fallback::Target`). The AMD and Apple atoms exist, and host
tests check that their layouts tile the instruction shape, but no kernel has been lowered or run
on those targets.

## Strategy

The decided approach is **one shared tile program per op**. A kernel such as `kernels/gemm.rs`
is written once against tile values and never branches on the vendor. What varies per target
is data the lowering and the op layer choose:

| Per target | Where |
|---|---|
| Matrix-core atoms and their layouts | `atoms`, `layout/atoms.rs` |
| Copy mechanisms (`cp.async`, `ldmatrix`, register staging) | `Target` flags, the emitter |
| Schedule templates | `schedule::Schedule` |
| Config candidate tables | `ops::config` |

A forked kernel program for one vendor is allowed only as a bounded exception.

## Not done

| Item | State |
|---|---|
| Hopper (sm_90a): wgmma, TMA, mbarrier, warp-specialized template | Not started; to be measured on remote H100 hardware |
| Blackwell (sm_100a): tcgen05, tensor memory | Not started; no B200 access |
| AMD CDNA: ping-pong template, MFMA 32×32, `buffer_load … lds` | Not started; to be measured on remote MI300X hardware |
| AMD RDNA, Apple | Atoms only; no tables, no emitted kernels |
| Warp roles, role barriers, raw asm statements | Recorded in the IR, rejected by the lowering |
| Convolution (implicit GEMM) | CUDA sm_80+ only (`ops::conv2d`); YOLO26 runs on it channels-last, and on AMD the op takes the graph until tk3 has RDNA tables |
| fp8/int8 weights, GEMV + argmax, persistent grid, attention backward | Not started |
