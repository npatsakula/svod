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

| Target | Atoms | Fills | Tables | State |
|---|---|---|---|---|
| CUDA sm_80+ (sm_86) | `mma.sync` m16n8k16, `ldmatrix` | `cp.async`, 2–3 stages | yes | Measured on an RTX 3060 |
| CUDA sm_90 (Hopper) | `mma.sync` m16n8k16, `ldmatrix` | `cp.async`, up to 227 KB shared | sm_80 tables | Compiled: every family assembles with `ptxas -arch=sm_90` and `sm_90a`; not run |
| AMD RDNA4 (gfx1200, gfx1201) | WMMA 16×16×16, 8 values per lane | register-staged, 2 stages | yes, unmeasured | Compiled to code objects; not run |
| AMD RDNA3 / RDNA3.5 (gfx1100–1102, gfx1151) | WMMA 16×16×16, replicated inputs | register-staged, 2 stages | yes, unmeasured | Compiled to code objects; not run |
| AMD CDNA3 / CDNA4 (gfx942, gfx950) | MFMA 16×16×16, wave64 | register-staged, 2 stages | yes, unmeasured | Compiled for gfx942; not run |
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
| Config candidate tables | `ops::config` |

A forked kernel program for one vendor is allowed only as a bounded exception.

## Not done

| Item | State |
|---|---|
| Hopper (sm_90a): wgmma, TMA, mbarrier, warp-specialized template | Not started; Hopper runs the `mma.sync` path (substrate items 5–8) |
| Blackwell (sm_100a): tcgen05, tensor memory | Not started; no B200 access |
| AMD CDNA: ping-pong template, MFMA 32×32, `buffer_load … lds` | Not started |
| AMD measurements | gfx1201 (RX 9070 XT) measured: GEMM, convolution and attention ahead of tk1 on Qwen3 and YOLO; CDNA compiled, not run |
| RDNA3 attention at d = 128 | The score tile still crosses shared memory there (the accumulator feeds neither operand) and the configs overrun the register file, so the graph runs it |
| Apple | Atoms only; no tables, no emitted kernels |
| Convolution (implicit GEMM) | Measured on sm_86, where YOLO26 runs on it channels-last (`ops::conv2d`); compiled only for AMD |
| Warp roles, role barriers, raw asm statements | Recorded in the IR, rejected by the lowering |
| fp8/int8 weights, GEMV + argmax, persistent grid, attention backward | Not started |
