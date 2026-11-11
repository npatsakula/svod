---
sidebar_label: Overview
---

# Tile Kernels (tk)

Svod's optimizer finds fast schedules for most of a model by searching over loop
transformations. Some kernels cannot be found that way. Flash attention is a recurrence: every
key block updates a running maximum and sum and rescales the accumulator, so there is no single
reduction to tile. A fast GEMM on a tensor-core GPU depends on a multi-stage `cp.async` ring,
`ldmatrix` fragment loads and a swizzled shared layout, which are not steps of a loop search.
Kernels like these are written by hand, and `svod-tk` ("tk") is the crate for writing them.

`svod-tk` implements the third version of the tile language ("tk3") and is the only tile-kernel
crate. Eight models run on the op layer:
Nemotron-3-Diarization, Whisper (the encoder, and the decoder step's attention in cache mode),
ModernBERT, BGE-M3/XLM-RoBERTa, Qwen3-Embedding, GigaAM, WavLM and YOLO26 (channels-last, its
3×3 convolutions on `ops::conv2d`).

## What tk is

| Piece | Module | What it does |
|---|---|---|
| Tile program | `ir`, `build` | A structured statement tree (`Let`, `Copy`, `Loop`, `Pipeline`, `If`, `Warps`) over tile values, recorded by a `Kernel` builder in program order. It is not a DAG. |
| Layouts | `layout`, `layouts` | F2 linear layouts (bit matrices) for every fragment and swizzle, assigned by inference. Kernel code never names a lane. |
| Atoms | `atoms` | Per-target matrix-core instructions with their operand layouts: `mma.sync` (CUDA), `wgmma` (sm_90), WMMA (RDNA3/3.5/4), MFMA 16×16×16, and 32×32×8 for attention (CDNA). |
| Schedule templates | `schedule` | Expand a `Pipeline` into `cp.async` groups, a register-staged ring of one or two slots, or two crews in ping-pong (CDNA GEMM). The author writes only the produce and consume bodies. |
| Lowering | `lower` | Inserts barriers from effects, expands the schedule, infers layouts and emits a pre-linearized instruction list (`Op::Linear`), so no toposort decides the order. |
| Kernels | `kernels` | GEMM with epilogues (its partial last wave split along K), implicit-GEMM convolution with split-K, flash attention, the attention prologue, LayerNorm/RMSNorm, and the nearest-neighbour search (argmin and top-k). Each is a spec struct in and a `Program` out; a kernel whose config the planner picks takes its spec admitted under that config (`Admitted`). |
| Launch | `launch` | Runs a program as a custom kernel in the lazy graph, with lowered bodies memoized. |
| Op layer | `ops` | `linear`, `conv2d`, `attention`, `heads`, `layer_norm`, `rms_norm`, `nearest`, `k_nearest`, … These always return a `Tensor`, using a kernel when one fits and the graph otherwise. |
| Planner | `ops::config` | Ranks each shape's configs from the device's measured costs and admits the config a spec runs under. |
| Interpreter | `interp` | Runs any tile program on the host to give reference numerics without a GPU. |
| Tune store | `tune` | Measures each op's candidates once per shape and device and keeps the winner on disk. |

```text
model code ──► svod_tk::ops          always a Tensor; picks a kernel and config or builds the graph op
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

| Target | Matrix core | Fills |
|---|---|---|
| CUDA sm_80+ | `mma.sync` m16n8k16, `ldmatrix` | `cp.async` |
| CUDA sm_90 | adds `wgmma` (GEMM and attention) | `cp.async` |
| AMD RDNA4 | WMMA 16×16×16 | registers, 1–2 slots |
| AMD RDNA3 / 3.5 | WMMA 16×16×16 | registers, 1–2 slots |
| AMD CDNA3 / 4 | MFMA 16×16×16, and 32×32×8 for attention; wave64 | registers, two crews in ping-pong |
| Apple | simdgroup 8×8×8 atom | — (no kernels) |

A target counts as measured only once
`targets::families_match_the_interpreter_on_the_device` passes on it. To see what a card reaches,
run `gemm_throughput_probe` and `attention_throughput_probe` on it
([Tuning](./tuning#probe-tests)): they print the tuner's pick in TFLOP/s.

:::note[Not in this release]
Hopper TMA and mbarrier producers (sm_90 runs `wgmma` fed by `cp.async`), Blackwell `tcgen05`,
32×32 MFMA for GEMM, Apple kernels, fp8, int8 beyond the nearest-neighbour search, attention backward, and
the convolution's image-staged form.
See [Portability](./portability#not-in-this-release).
:::

## Reading order

1. [Your First Kernel](./first-kernel): the builder, walking through the GEMM.
2. [Layouts and Lowering](./layouts-and-lowering): what happens between the program and the device.
3. [Kernel Library](./kernel-library): the kernels and their configs.
4. [Op Layer](./op-layer): the API models call.
5. [Tuning](./tuning), [Testing and Debugging](./testing-and-debugging), [Portability](./portability).
