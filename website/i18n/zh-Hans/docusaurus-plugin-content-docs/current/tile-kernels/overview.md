---
sidebar_label: 概览
---

# 为什么要手写内核？

Svod 的一切都围绕自动化展开。你搭好一张惰性图，调用 `realize()`，优化器便会替每个循环决定如何分块、向量化、并行化；若再配上 [beam search](../architecture/optimizations/kernel-search)，它甚至会编译并实测上百个候选调度，从中挑出最快的一个。整个过程你一行循环都不必写。

那么 Svod 为什么还要专门提供一个 crate，即 `svod-tk`（简称 `tk`），它的全部职责恰恰是让你亲手编写 GPU 内核？

因为有些内核，靠在循环变换空间里搜索是发现不了的。优化器能做的动作无非是「拿这个规约去分块、展开、塞进共享内存」。对 matmul、对融合的前馈块、对 layernorm 而言，这些动作已经够用。但它们对付不了 Flash Attention，因为后者的数学本质是一个*递推*：每读入一块 key，就要刷新当前的最大值和累加和，并顺势重新缩放累加器。这里根本没有单一的 `REDUCE` 供你分块，循环体本身就依赖上一轮迭代的结果。无论怎样调换轴的顺序，都变不出这种结构。

这类内核，只能把算法亲手写出来。`tk` 正是为此而生，而且全程不必走出编译器。

---

## `tk` 是构建器，不是后端

需要手写内核时，最容易掉进去的陷阱是再开一条代码路径：搞一个小型 GPU DSL，让它发射自己的汇编，再独立于其他一切单独启动。结果你手里多出了两个编译器、两个调试器、两套心智模型。

`tk` 偏不这么做。用它自己的话说，它是「一个轻量的即时构建器，而非后端」。当你用 `tk` 编写内核时，它并不发射机器码，而是发射 Svod 其余部分早已通晓的**同一套 UOp IR**：显式的 `RANGE` 循环、`INDEX`/`STORE` 内存操作、`WMMA` 矩阵核心操作，正是 [一个 IR 统治一切](../architecture/ir-design) 所描述的那种中间表示。

这意味着手写的 `tk` 内核与自动调优的图内核属于*同一类对象*：同一张 UOp DAG 里的两个子图，由同一个渲染器渲染，由同一个运行时执行。[向 IR 中编写](./lowering) 会把其中的门道讲透。

---

## `tk` 的三副面孔

视你的身份与目的而定，`tk` 会呈现以下三种接口之一（它们全部从 `tk/src/lib.rs` 重新导出）：

| 面孔 | 你是…… | 你接触的东西 |
|------|----------|----------------|
| **USE** | 只想拿到一个快内核的应用作者 | `flash_attention`、带融合 `Epilogue` 的 `gemm_nt`、`rms_norm` / `add_rms_norm`、`single_query_attention`、`knn`、`kmeans_assign`、方阵 `matmul`，它们返回惰性 `Tensor`，无需任何内核知识 |
| **AUTHOR** | 正在编写一个新的 tile 内核 | `Kernel` / `Group` / `Loop` 构建器、`ArchCaps` 与 `FragRole`、各 tile 类型（`GL`/`ST`/`RT`/`RV`）、`MoveIdx`、`graph_launch` 与 `launch_custom` |
| **DEBUG** | 想隔离测试或基准测试某个内核 | `run_kernel`、`compile_kernel`、`CompiledLaunch`、结构化的 `KernelFingerprint`、`tune` 存储 |

对多数读者来说，USE 面孔才是关键：`flash_attention(q, k, v)` 还给你一个普通 `Tensor`，它像别的张量一样参与惰性图，你压根看不到任何 tile。[内核库](./kernel-library) 列出了全部内核，[什么是分块](./tiling) 会揭开 AUTHOR 面孔，[调试](./debugging) 则讲 DEBUG。

---

## 目标平台

`tk` 内核面向某个矩阵核心家族构建，每个内核都以 `ArchSet`（`tk/src/target.rs`）声明它已验证过的型号。在其他任何设备上，它的启动器返回 `Ok(None)`，调用方随即回退到图路径。

| 家族 | 型号 | 矩阵操作 | Wave |
|---|---|---|---|
| AMD CDNA3 | gfx942 | MFMA | 64 lanes |
| AMD RDNA3.5 | gfx1151 | gfx11 WMMA | 32 lanes |
| AMD RDNA4 | gfx1200, gfx1201 | gfx12 WMMA | 32 lanes |
| NVIDIA | sm_80 及更新 | `mma.sync.m16n8k16` | 32 lanes |
| Apple | Apple7 及更新（仅 flash attention 与 `matmul`） | `simdgroup_matrix` 8×8 | 32 lanes |

输入为 bf16 或 f16，累加为 f32。哪个内核在哪里运行，见 [内核库](./kernel-library) 中的表格；同一份内核体为何能在所有这些平台上运行，见 [布局与 wave 宽度](./wave-portability)。

---

## 何时手写，何时交给 BEAM

规则只有一条，而它直接来自一个问题：*BEAM 究竟在搜索什么*。

BEAM，连同它兜底时所用的启发式优化器，搜索的是某个*固定*计算的**调度**空间。给定一张内核数据流图，它们尝试各种分块、向量化、展开、并行化、经共享内存分阶段、以及映射到矩阵核心的方式（即 `OptOps` 动作：`UPCAST`、`UNROLL`、`LOCAL`、`GROUP`、`TC`……）。它们唯独不会改的，是*算什么*：图的节点，那些加法、乘法、规约，全是固定的，可调的只有它们的排布方式。

所以：

> 如果一个内核只是缺一份好**调度**而数据流固定，那就交给 BEAM 去找。如果它需要一个与朴素实现不同的**算法**，一种现有操作无论怎么重排都凑不出来的东西，
> 或者需要一次跨越调度器所保留的内核边界的**融合**，那你就只能亲手写。

| 内核的属性 | 由谁构建 | 示例 |
|------------------------|----------|----------|
| **数据流固定**：在矩形迭代空间上做逐元素操作和规约，可调的只有*调度*（分块、向量化、数据布局、矩阵核心映射） | 图操作 + **BEAM** | 普通 matmul、前馈块、layernorm、softmax |
| **需要重新表述算法**：带循环依赖的递推，或重新组织的数值流程，朴素操作怎么重排都产生不出来 | **在 `tk` 中手写** | Flash Attention（在线 softmax）；单查询 attention（在流式传入的缓存上做单遍 softmax）；k-NN 与 k-means 分配——把一次交叉项 WMMA 与一个在流式 tile 上滚动进行的 top-K / argmin 相融合，于是完整的 `[N, M]` 距离矩阵从头至尾都无需构造 |
| **需要调度器无法表达的融合**：把额外工作折进矩阵核心内核自身的写回，或让残差流只写一次 | **在 `tk` 中手写** | 带残差加法或 SwiGLU `Epilogue` 的 `gemm_nt`；`add_rms_norm` |

### BEAM 够不到的地方

朴素的 attention 会构造出整个 `N×N` 的得分矩阵，对它取全局 softmax，再乘以 `V`。BEAM 固然能给这套流程分块、向量化，但它依然得把完整的得分矩阵物化出来，而这正是 Flash Attention 要规避的开销。

快版本从不构造那个矩阵。它流式扫过一块块 key，维护当前的最大值与累加和，并在每块到来时重新缩放输出，这就是在线 softmax。它不是把朴素计算重新调度一遍，而是一种带循环依赖的全新数据流：每一块都要读取上一块写下的状态。任何 `UPCAST`/`UNROLL`/`TC` 序列都引入不了递推，因此在线 softmax 落在 BEAM 的搜索空间之外。这道坎是算法上的，而非调度上的，而它正是 `tk` 要填的空白。

第三行是一道更窄的坎。图中的线性层是一个 BEAM 能调得很好的矩阵核心内核，但紧随其后的残差加法，或融合 gate/up 投影之后的 `silu(gate) · up`，都是对输出的第二遍处理，调度器不会把它折进 GEMM 的 epilogue。`gemm_nt` 把它折了进去，于是中间结果从不写出。第一行中的方阵 `matmul` 又是另一回事：它是这门 DSL 的性能风向标，并非生产用的 matmul（后者走的是图这条路）。

:::tip[面向 GPU 专家]
手写内核与 BEAM 调优内核之间的结构差异，仅在 `SINK` UOp 的 `KernelInfo` 上一个字段：图内核令 `opts_to_apply: None`，`tk` 内核则设为 `Some(vec![])`。同一套 IR、同一条流水线，只差一个标记。[向 IR 中编写](./lowering) 会端到端地追踪这一点。
:::

---

## 本节走向

本节余下内容从硬件难题一路铺陈到设计对比：

1. **[FLOPS 藏在哪里](./where-flops-hide)**：为什么矩阵核心很难喂饱，以及每个快内核都得跨过的那几个瓶颈。
2. **[什么是分块](./tiling)**：回应这些瓶颈的抽象，以及 `tk` 如何在类型系统里表示 tile。
3. **[向 IR 中编写](./lowering)**：一个 `tk` 内核如何变成 UOp 并融入惰性图。
4. **[编写一个内核](./first-kernel)**：手把手编写并运行最简单的内核。
5. **[构建器 API](./builder-reference)**：AUTHOR 面孔的每个类型与方法。
6. **[布局与 wave 宽度](./wave-portability)**：让同一个内核在五种 fragment 布局和两种 wave 宽度上都保持正确。
7. **[内核库](./kernel-library)**：随附的内核、它们的契约，以及模型如何使用它们。
8. **[Flash Attention](./flash-attention)**：促成这一切的实战范例。
9. **[自动调优](./tuning)**：内核的 tile 如何在首次使用时测量并缓存。
10. **[调试](./debugging)**：手动运行并验证内核。
11. **[剖析与基准测试](./profiling)**：分层 profiler 与 criterion 集成，适用于任何 `Tensor` 或 `ExecutionPlan`。
12. **[tk、HipKittens 与 CuTile 对比](./comparison)**：这套设计在整个版图中的位置。
