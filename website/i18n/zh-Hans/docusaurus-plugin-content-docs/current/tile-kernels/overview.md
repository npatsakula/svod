---
sidebar_label: 概览
---

# Tile 内核 (tk3) {#tile-kernels-tk3}

Svod 的优化器通过搜索循环变换，为模型的大部分计算找到快速的调度。但有些内核无法用这种方式找到。Flash attention 是一个递推：每个键块都要更新运行中的最大值与和，并重新缩放累加器，因此不存在一个可以直接分块的归约。张量核心 GPU 上的快速 GEMM 依赖多级 `cp.async` 环形缓冲、`ldmatrix` 片段加载以及带 swizzle 的共享内存布局，这些都不是循环搜索中的步骤。这类内核需要手写，而 `svod-tk3`（“tk3”）就是编写它们的 crate。

tk3 是唯一的 tile 内核 crate。它取代了 tk1（`svod-tk`），tk1 已连同其 k-means、kNN 和 Apple 内核一起移除，这些内核均未移植。有八个模型运行在算子层上：Nemotron-3-Diarization、Whisper（编码器，以及缓存模式下解码器步骤中的注意力）、ModernBERT、BGE-M3/XLM-RoBERTa、Qwen3-Embedding、GigaAM、WavLM 和 YOLO26（通道在后布局，其 3×3 卷积使用 `ops::conv2d`）。

## tk3 是什么 {#what-tk3-is}

| 组成部分 | 模块 | 作用 |
|---|---|---|
| Tile 程序 | `ir`, `build` | 基于 tile 值的结构化语句树（`Let`、`Copy`、`Loop`、`Pipeline`、`If`、`Warps`），由 `Kernel` 构建器按程序顺序记录。它不是 DAG。 |
| 布局 | `layout`, `layouts` | 用 F2 线性布局（比特矩阵）表示每种片段和 swizzle，由推断分配。内核代码从不指名某个 lane。 |
| 原子 | `atoms` | 各目标的矩阵核心指令及其操作数布局：`mma.sync`（CUDA）、`wgmma`（sm_90）、WMMA（RDNA3/3.5/4）、MFMA 16×16×16（CDNA）。 |
| 调度模板 | `schedule` | 把 `Pipeline` 展开为 `cp.async` 组、一或两个槽位的寄存器暂存环形缓冲，或以乒乓方式交替的两个工作组（CDNA GEMM）。作者只需编写生产与消费两部分的主体。 |
| 降级 | `lower` | 根据效果插入屏障、展开调度、推断布局，并输出预线性化的指令列表（`Op::Linear`），因此顺序不由任何拓扑排序决定。 |
| 内核 | `kernels` | 带尾处理（epilogue）的 GEMM、带 split-K 的隐式 GEMM 3×3 卷积、flash attention、注意力前处理（prologue）、LayerNorm/RMSNorm。每个内核输入一个 spec 结构体，输出一个 `Program`。 |
| 启动 | `launch` | 把程序作为惰性图中的自定义内核运行，降级后的主体会被记忆化缓存。 |
| 算子层 | `ops` | `linear`、`conv2d`、`attention`、`heads`、`layer_norm`、`rms_norm`……它们总是返回 `Tensor`：有合适的内核时使用内核，否则使用计算图。 |
| 规划器 | `ops::config` | 依据设备测得的代价，为每种形状的配置排序。 |
| 解释器 | `interp` | 在主机上运行任意 tile 程序，无需 GPU 即可给出参考数值。 |
| 调优存储 | `tune` | 对每种形状和设备，把每个算子的候选测量一次，并把胜者保存在磁盘上。 |

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

## 目标 {#targets}

| 目标 | 矩阵核心 | 填充方式 | 测量所用硬件 | GEMM 4096³ bf16，TFLOP/s | 注意力 d64，TFLOP/s |
|---|---|---|---|---|---|
| CUDA sm_80+ | `mma.sync` m16n8k16、`ldmatrix` | `cp.async` | RTX 3060 (sm_86)；RTX 5070、RTX 5090 (sm_120) | 25.9；65.7（cuBLAS NN 67.4）；204.8（cuBLAS NN 218.2） | 24.8；62.9；188.5 |
| CUDA sm_90 | 增加 `wgmma`（GEMM 与注意力） | `cp.async` | H100 PCIe | 423.7（cuBLASLt 410–524） | 254.9（FA3 270.1） |
| AMD RDNA4 | WMMA 16×16×16 | 寄存器，1–2 个槽位 | RX 9070 XT (gfx1201) | 148.8（rocBLAS 80） | 117.3 |
| AMD RDNA3 / 3.5 | WMMA 16×16×16 | 寄存器，1–2 个槽位 | Strix Halo (gfx1151)；gfx1100 已编译，未运行 | 29.7（rocBLAS NT 32.5） | 26.2 |
| AMD CDNA3 / 4 | MFMA 16×16×16，wave64 | 寄存器，两个工作组乒乓交替 | MI300X (gfx942)；gfx950 未运行 | 560.0（hipBLASLt 615） | 281.4 |
| Apple | simdgroup 8×8×8 原子 | — | 无内核 | — | — |

每个数字都是调优器在该卡上的选择，于 2026-10-17 用 `gemm_throughput_probe` 和 `attention_throughput_probe`（B 4、H 8、T 2048）测得，提交为 55fb615b，H100（251276d9）和 MI300X（fc5cc954）除外。括号中的厂商参考值在同一台机器上计时。只有当 `targets::families_match_the_interpreter_on_the_device` 在某个目标上通过后，该目标才算已测量。

在 RTX 3060 上，对 8192×1024 bf16 做 RMSNorm 的带宽为 325 GB/s（该卡上限为 360）；Nemotron-3-Diarization（bf16，四步）用 1064 次调度耗费 45.82 ms GPU 时间，而在 tk3 之前需要 2292 次调度。

:::note[本版本未包含]
Hopper TMA 与 mbarrier 生产者（sm_90 运行由 `cp.async` 供数的 `wgmma`）、Blackwell `tcgen05`、32×32 MFMA、Apple 内核、fp8/int8、注意力反向传播，以及卷积的图像暂存形式。参见[可移植性](./portability#not-in-this-release)。
:::

## 阅读顺序 {#reading-order}

1. [第一个内核](./first-kernel)：构建器，逐步讲解 GEMM。
2. [布局与降级](./layouts-and-lowering)：从程序到设备之间发生了什么。
3. [内核库](./kernel-library)：各个内核及其配置。
4. [算子层](./op-layer)：模型调用的 API。
5. [调优](./tuning)、[测试与调试](./testing-and-debugging)、[可移植性](./portability)。
