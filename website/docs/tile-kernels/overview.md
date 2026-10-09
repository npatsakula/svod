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

tk3 succeeds the older `svod-tk` crate (tk1). The transformer models run their hand kernels on the
op layer described here, and so does YOLO26, whose convolutions are implicit GEMMs (`ops::conv2d`).

## What tk3 is

| Piece | Module | What it does |
|---|---|---|
| Tile program | `ir`, `build` | A structured statement tree (`Let`, `Copy`, `Loop`, `Pipeline`, `If`) over tile values. It is built by a recording `Kernel` builder in program order and is not a DAG. |
| Layouts | `layout`, `layouts` | F2 linear layouts (bit matrices) for every fragment and swizzle, assigned by inference. Kernel code never names a lane. |
| Atoms | `atoms` | Per-target matrix-core instructions that carry their operand layouts (`mma.sync` on CUDA). |
| Schedule templates | `schedule` | Expand a `Pipeline` statement into loops, `cp.async` groups and barriers. The author writes only the produce and consume bodies. |
| Lowering | `lower` | Expands the schedule, inserts barriers from effects, infers layouts and emits a pre-linearized instruction list (`Op::Linear`), so no toposort decides the order. |
| Kernels | `kernels` | GEMM with epilogues, flash attention, the attention prologue, LayerNorm/RMSNorm. Each is a spec struct in and a `Program` out. |
| Launch | `launch` | Runs a program as a custom kernel in the lazy graph, with lowered bodies memoized. |
| Op layer | `ops` | `linear`, `attention`, `heads`, `layer_norm`, `rms_norm`, … These always return a `Tensor`, using a kernel when one fits and the graph otherwise. |
| Interpreter | `interp` | Runs any tile program on the host to give reference numerics without a GPU. |
| Tune store | `tune` | Measures each op's config candidates once per shape and device and keeps the winner on disk. |

```text
model code ──► svod_tk3::ops          always a Tensor; picks a kernel and config or builds the graph op
                   │
                   ▼
              kernels::*               spec ─► tile Program (builder, program order)
                   │
                   ▼
              lower::lower             schedule template ─► operand loads ─► barriers
                   │                   ─► layout inference ─► emission
                   ▼
              Op::Linear program  ──►  custom kernel in the lazy graph ─► LLVM ─► device
```

## Status, measured on an RTX 3060 (sm_86)

| What | Result |
|---|---|
| bf16 GEMM 4096³ | 25.4 TFLOP/s, equal to tk1. The `mma.sync` bf16→f32 ceiling of this card is 27.9 |
| Flash attention forward | At or above tk1 on every throughput probe (head dims 64 and 128, causal and not) |
| LayerNorm / RMSNorm | 92% of memory bandwidth |
| Nemotron-3-Diarization (bf16) | 68.2 → 49–53 ms per four steps, 2292 → 1304 dispatches, parity kept |

Nemotron uses the op layer for all of its projections, norms and attention. The Whisper
encoder, ModernBERT, GigaAM, XLM-R/BGE-M3 and Qwen3 call it for projections and norms.

:::note[Not done yet]
Decode attention in the models, convolution kernels, and any target other than NVIDIA sm_80+
(Hopper, Blackwell and AMD CDNA/RDNA paths) are not implemented. See
[Portability](./portability).
:::

## Reading order

1. [Your First Kernel](./first-kernel): the builder, walking through the GEMM.
2. [Layouts and Lowering](./layouts-and-lowering): what happens between the program and the device.
3. [Kernel Library](./kernel-library): the kernels and their configs.
4. [Op Layer](./op-layer): the API models call.
5. [Tuning](./tuning), [Testing and Debugging](./testing-and-debugging), [Portability](./portability).
