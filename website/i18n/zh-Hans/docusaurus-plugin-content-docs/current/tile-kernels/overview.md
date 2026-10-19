---
sidebar_label: 概览
---

# Tile 内核 (tk3)

Svod 的优化器通过搜索循环变换，为模型的大部分计算找到快速的调度。但有些内核无法用这种方式找到。Flash attention 是一个递推：每个键块都要更新运行中的最大值与和，并重新缩放累加器，因此不存在一个可以直接分块的归约。张量核心 GPU 上的快速 GEMM 依赖多级 `cp.async` 环形缓冲、`ldmatrix` 片段加载以及带 swizzle 的共享内存布局，这些都不是循环搜索中的步骤。这类内核需要手写，而 `svod-tk3`（“tk3”）就是编写它们的 crate。

tk3 取代了较早的 `svod-tk` crate（tk1）。随着模型迁移到本文介绍的算子层，tk1 正在被淘汰。

## tk3 是什么 {#what-tk3-is}

| 组成部分 | 模块 | 作用 |
|---|---|---|
| Tile 程序 | `ir`, `build` | 基于 tile 值的结构化语句树（`Let`、`Copy`、`Loop`、`Pipeline`、`If`）。它由记录式 `Kernel` 构建器按程序顺序构建，不是 DAG。 |
| 布局 | `layout`, `layouts` | 用 F2 线性布局（比特矩阵）表示每种片段和 swizzle，由推断分配。内核代码从不指名某个 lane。 |
| 原子 | `atoms` | 各目标的矩阵核心指令，自带操作数布局（CUDA 上为 `mma.sync`）。 |
| 调度模板 | `schedule` | 把 `Pipeline` 语句展开为循环、`cp.async` 组和屏障。作者只需编写生产与消费两部分的主体。 |
| 降级 | `lower` | 展开调度、根据效果插入屏障、推断布局，并输出预线性化的指令列表（`Op::Linear`），因此顺序不由任何拓扑排序决定。 |
| 内核 | `kernels` | 带尾处理（epilogue）的 GEMM、flash attention、注意力前处理（prologue）、LayerNorm/RMSNorm。每个内核输入一个 spec 结构体，输出一个 `Program`。 |
| 启动 | `launch` | 把程序作为惰性图中的自定义内核运行，降级后的主体会被记忆化缓存。 |
| 算子层 | `ops` | `linear`、`attention`、`heads`、`layer_norm`、`rms_norm`……它们总是返回 `Tensor`：有合适的内核时使用内核，否则使用计算图。 |
| 解释器 | `interp` | 在主机上运行任意 tile 程序，无需 GPU 即可给出参考数值。 |
| 调优存储 | `tune` | 对每种形状和设备，把每个算子的配置候选测量一次，并把胜者保存在磁盘上。 |

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

## 现状：在 RTX 3060 (sm_86) 上测得 {#status-measured-on-an-rtx-3060-sm_86}

| 项目 | 结果 |
|---|---|
| bf16 GEMM 4096³ | 25.4 TFLOP/s，与 tk1 持平。这张卡上 `mma.sync` bf16→f32 的上限是 27.9 |
| Flash attention 前向 | 在所有吞吐量探针上都不低于 tk1（head 维度 64 和 128，因果与非因果） |
| LayerNorm / RMSNorm | 达到内存带宽的 92% |
| Nemotron-3-Diarization (bf16) | 每四步 68.2 → 49–53 ms，调度次数 2292 → 1304，保持一致性 |

Nemotron 的所有投影、归一化和注意力都使用算子层。Whisper 编码器、ModernBERT、GigaAM、XLM-R/BGE-M3 和 Qwen3 在投影和归一化中调用它。

:::note[尚未完成]
模型中的解码注意力、卷积内核，以及 NVIDIA sm_80+ 之外的任何目标（Hopper、Blackwell 和 AMD CDNA/RDNA 路径）都尚未实现。参见[可移植性](./portability)。
:::

## 阅读顺序 {#reading-order}

1. [第一个内核](./first-kernel)：构建器，逐步讲解 GEMM。
2. [布局与降级](./layouts-and-lowering)：从程序到设备之间发生了什么。
3. [内核库](./kernel-library)：各个内核及其配置。
4. [算子层](./op-layer)：模型调用的 API。
5. [调优](./tuning)、[测试与调试](./testing-and-debugging)、[可移植性](./portability)。
