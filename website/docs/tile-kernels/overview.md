---
sidebar_label: Overview
---

# Tile Kernels (tk3)

Svod's optimizer finds fast schedules for most of a model by searching over loop
transformations. Some kernels cannot be found that way. Flash attention is a recurrence: every
key block updates a running maximum and sum and rescales the accumulator, so there is no single
reduction to tile. A fast GEMM on a tensor-core GPU depends on a multi-stage `cp.async` ring,
`ldmatrix` fragment loads and a swizzled shared layout, which are not steps of a loop search.
Kernels like these are written by hand, and `svod-tk3` ("tk3") is the crate for writing them.

tk3 replaces tk1 (`svod-tk`) as the source of the models' hand kernels; no model calls tk1 any
more, and `tk/` stays in the tree. Eight models run on the op layer: Nemotron-3-Diarization,
Whisper (the encoder, and the decoder step's attention in cache mode), ModernBERT,
BGE-M3/XLM-RoBERTa, Qwen3-Embedding, GigaAM, WavLM and YOLO26 (channels-last, its 3×3
convolutions on `ops::conv2d`).

## What tk3 is

| Piece | Module | What it does |
|---|---|---|
| Tile program | `ir`, `build` | A structured statement tree (`Let`, `Copy`, `Loop`, `Pipeline`, `If`, `Warps`) over tile values, recorded by a `Kernel` builder in program order. It is not a DAG. |
| Layouts | `layout`, `layouts` | F2 linear layouts (bit matrices) for every fragment and swizzle, assigned by inference. Kernel code never names a lane. |
| Atoms | `atoms` | Per-target matrix-core instructions with their operand layouts: `mma.sync` (CUDA), `wgmma` (sm_90), WMMA (RDNA3/3.5/4), MFMA 16×16×16 (CDNA). |
| Schedule templates | `schedule` | Expand a `Pipeline` into `cp.async` groups, a register-staged ring of one or two slots, or two crews in ping-pong (CDNA GEMM). The author writes only the produce and consume bodies. |
| Lowering | `lower` | Inserts barriers from effects, expands the schedule, infers layouts and emits a pre-linearized instruction list (`Op::Linear`), so no toposort decides the order. |
| Kernels | `kernels` | GEMM with epilogues, implicit-GEMM 3×3 convolution with split-K, flash attention, the attention prologue, LayerNorm/RMSNorm. Each is a spec struct in and a `Program` out. |
| Launch | `launch` | Runs a program as a custom kernel in the lazy graph, with lowered bodies memoized. |
| Op layer | `ops` | `linear`, `conv2d`, `attention`, `heads`, `layer_norm`, `rms_norm`, … These always return a `Tensor`, using a kernel when one fits and the graph otherwise. |
| Planner | `ops::config` | Ranks each shape's configs from the device's measured costs. |
| Interpreter | `interp` | Runs any tile program on the host to give reference numerics without a GPU. |
| Tune store | `tune` | Measures each op's candidates once per shape and device and keeps the winner on disk. |

```text
model code ──► svod_tk3::ops          always a Tensor; picks a kernel and config or builds the graph op
                   │
                   ▼
              kernels::*               spec ─► tile Program (builder, program order)
                   │
                   ▼
              lower::lower             operand loads ─► barriers ─► schedule template
                   │                   ─► layout inference ─► emission
                   ▼
              Op::Linear program  ──►  custom kernel in the lazy graph ─► LLVM ─► device
```

## Targets

| Target | Matrix core | Fills | Measured on | GEMM 4096³ bf16, TFLOP/s | Attention d64, TFLOP/s |
|---|---|---|---|---|---|
| CUDA sm_80+ | `mma.sync` m16n8k16, `ldmatrix` | `cp.async` | RTX 3060 (sm_86); RTX 5070, RTX 5090 (sm_120) | 25.9; 65.7 (cuBLAS NN 67.4); 204.8 (cuBLAS NN 218.2) | 24.8; 62.9; 188.5 |
| CUDA sm_90 | adds `wgmma` (GEMM and attention) | `cp.async` | H100 PCIe | 423.7 (cuBLASLt 410–524) | 254.9 (FA3 270.1) |
| AMD RDNA4 | WMMA 16×16×16 | registers, 1–2 slots | RX 9070 XT (gfx1201) | 148.8 (rocBLAS 80) | 117.3 |
| AMD RDNA3 / 3.5 | WMMA 16×16×16 | registers, 1–2 slots | Strix Halo (gfx1151); gfx1100 compiled, not run | 29.7 (rocBLAS NT 32.5) | 26.2 |
| AMD CDNA3 / 4 | MFMA 16×16×16, wave64 | registers, two crews in ping-pong | MI300X (gfx942); gfx950 not run | 560.0 (hipBLASLt 615) | 281.4 |
| Apple | simdgroup 8×8×8 atom | — | no kernels | — | — |

Each figure is the tuner's pick on that card, measured on 2026-10-17 with
`gemm_throughput_probe` and `attention_throughput_probe` (B 4, H 8, T 2048), at commit 55fb615b
except the H100 (251276d9) and the MI300X (fc5cc954). Vendor references in parentheses were timed
on the same machine. A target counts as measured only once
`targets::families_match_the_interpreter_on_the_device` passes on it.

On the RTX 3060, RMSNorm over 8192×1024 bf16 moves 325 GB/s of the card's 360, and
Nemotron-3-Diarization (bf16, four steps) takes 45.82 ms of GPU time in 1064 dispatches, against
2292 dispatches before tk3.

:::note[Not in this release]
Hopper TMA and mbarrier producers (sm_90 runs `wgmma` fed by `cp.async`), Blackwell `tcgen05`,
32×32 MFMA, Apple kernels, fp8/int8, attention backward, and the convolution's image-staged form.
See [Portability](./portability#not-in-this-release).
:::

## Reading order

1. [Your First Kernel](./first-kernel): the builder, walking through the GEMM.
2. [Layouts and Lowering](./layouts-and-lowering): what happens between the program and the device.
3. [Kernel Library](./kernel-library): the kernels and their configs.
4. [Op Layer](./op-layer): the API models call.
5. [Tuning](./tuning), [Testing and Debugging](./testing-and-debugging), [Portability](./portability).
