---
sidebar_label: 布局与 wave 宽度
---

# 让同一个内核在五种布局上保持正确

这里有个 NVIDIA 硬件根本给不了你的 bug。你写了一个 tile 内核，在 CDNA 数据中心 GPU 上一测，完美无缺。换到一台 RDNA 笔记本 APU 上跑*同一个*内核，结果却是一堆垃圾数字，没崩溃、没报错，就是错。代码看上去没有任何不同。CUDA 躲过了这个特定的陷阱——warp 处处都是 32 个 lane——却躲不过它背后的成因：片段布局依然不同，所以正是同一层间接把内核带上了 NVIDIA，也带上了 Apple。

[什么是分块](./tiling) 引入了片段和基于角色的选择；本章解释那层间接为什么非有不可。罪魁祸首是**波前大小**，以及藏在它下面的片段 **lane 映射**；能否干净地应对这两者，正是「只能在一种芯片上工作的 tile 库」与「真正可移植的 tile 库」之间的分水岭。

---

## 32 与 64 之分

波前（NVIDIA 叫「warp」，Apple 叫「SIMD group」）是一群锁步执行的 lane。AMD 上有两种大小，Svod 两者都瞄准，此外还有 NVIDIA 与 Apple 的：

| 家族 | 型号 | 矩阵操作 | 波前 | 片段 | 每 lane |
|---|---|---|---|---|---|
| **CDNA3** | gfx942 | MFMA | 64 | 16×16，每个角色都是 `Strided { stride: 4 }` | 4 |
| **RDNA3** | gfx1151 | gfx11 WMMA | 32 | 16×16，`Interleaved` 累加器，`Strided { stride: 0 }` 操作数在两个半 wave 间复制 | 累加器 8 / 操作数 16 |
| **RDNA4** | gfx1200, gfx1201 | gfx12 WMMA | 32 | 16×16，每个角色都是 `Strided { stride: 8 }` | 8 |
| **CUDA** | sm_80+ | `mma.sync.m16n8k16` | 32 | 16×16 拆为两个 `m16n8` 半块，每个角色都是 `MmaSync` | 8 |
| **Metal** | Apple7+ | `simdgroup_matrix<T, 8, 8>` | 32 | 8×8，`SimdgroupMatrix`（B 与累加器），`SimdgroupMatrixT`（A） | 2 |

（这张表是 DSL 会解析的那组布局，即 `tk/src/arch.rs` 中的 `ArchCaps::frag` 与 `tk/src/tiles.rs` 中的常量。在它之上，每个内核还各自声明自己的 `ArchSet`；逐内核的矩阵见 [内核库](./kernel-library)。）

就这么一个数字，却牵动着一切。一个 `16×16` tile 有 256 个元素：摊到 64 个 lane 上，每 lane 4 个；摊到 32 个 lane 上，每 lane 8 个——RDNA3 除外，那里操作数被复制，每个 lane 持有 16 个。不同的 lane 持有不同的元素。于是：

- tile 的**寄存器布局**不同，
- 矩阵指令所要的**操作数布局**不同，
- 而任何**跨 lane 规约**（softmax 与 layernorm 的核心）都有着不同的步数和不同的兄弟模式。

一个硬编码了「有 64 个 lane，gather lane 16、32、48 来规约」的内核，在 32-lane 机器上算出的只是一个*部分*规约，并悄无声息地返回错误的值。

---

## 对策：索要角色，而非形状

`tk` 的答案是加一层间接。内核从不写下「16×16，每 lane 4 个元素」这样一个具体的片段形状。它索要的是一个**角色**，再交由架构能力去解析：

```text
   kernel says:  "I need an accumulator fragment"   (FragRole::Accumulator)
                          │
                          ▼
   ArchCaps::frag(role)   ── on CDNA  ──▶  RT_16X16          (wave64, 4/lane)
                          ├─ on gfx11 ──▶  RT_16X16_W32_ACC  (even/odd rows, 8/lane)
                          ├─ on gfx12 ──▶  RT_16X16_GFX12    (strided, 8/lane)
                          ├─ on CUDA  ──▶  RT_16X16_MMA      (two m16n8 halves, 8/lane)
                          └─ on Metal ──▶  RT_8X8_SIMD       (2/lane)
```

这些角色是 `FragRole::{Accumulator, Operand, OperandB, AccumulatorT}`，解析器则是 `ArchCaps::frag(role)`（内核通过 `ker.frag(role)` 触到它，或经由快捷方式 `ker.acc` / `ker.operand` / `ker.operand_b` / `ker.acc_t`）。内核作者只管写「accumulator」和「operand」；至于*物理*布局（每 lane 元素数、interleave 映射、复制），则替目标平台填补进去。CDNA 与 gfx12 把每个角色都解析到同一个形状；gfx11 把累加器、它的转置与复制式操作数分开；CUDA 把每个角色都解析到两半式映射；Metal 给 A 操作数一个翻转的映射，因为它的核心直接按 lane 映射计算 `D = A·B`，而 tk 的累加器是列主序的。凡是 tk 压根没有对应表的地方（Ampere 之前的 CUDA、Apple7 之前的 Metal）则是 `None`，好让矩阵核心内核在 `ker.frag` 处大声失败，而不是渲染出一个错误的布局。写一次，五者都能跑。

HipKittens 学到的也是这一课（见 [tk、HipKittens 与 CuTile 对比](./comparison)）：它的 tile 类型以单个编译期 `WARP_THREADS` 常量为键（CDNA 构建里是 `64`），所以换一种 wave 宽度就意味着换一份该库的构建。`tk` 则把这一套折叠成一个运行时解析的 `ArchCaps`。

---

## 一个它真抓住过的 bug

这层间接之所以存在，并非纸上谈兵。早期 `tk` 的一个跨 lane 全规约，即用来把一个值在一个 wave 上求和的 `shuffle_xor` 原语，当初是用硬编码的 wave64 规约树写的。在 RDNA 的 32-lane wave 上，它对那些根本不参与的 lane 做规约，对 attention 所依赖的那种 softmax 式规约算出了错误的和。修复办法是改为基于解析出的片段来驱动规约，而不是一个常量。`tk/src/group/shuffle.rs` 里的混洗原语读取 `caps.wave_size`；规约读取片段的 `LaneMap`；这类 bug 从设计上就被消除了。

:::tip[面向 GPU 专家]
承担布局相关大部分分量的是两样东西，它们都从解析出的片段的 `LaneMap`（`tk/src/layout.rs`）读取，而从不读取常量：

- **规约树。** `LaneMap::tree(wave_size)` 是片段规约在 lane 内折叠之后的跨 lane 收尾。AMD 的映射用 `ds_bpermute` gather 每个兄弟 lane 组的*原始*部分和，wave64 上偏移为 `[16, 32, 48]`，wave32 上为 `[16]`；`MmaSync` 用 `shfl.bfly` 在掩码 `[1, 2]` 上对*滚动*值做蝶形（一个 lane 的八个元素跨越两行，所以它保留 `LaneMap::slots() == 2` 个值，折叠在 4-lane 的 quad 内完成）；`SimdgroupMatrix` 则在 `[1, 8]` 上做蝶形。`tk/src/group/reduce.rs` 遍历交给它的无论哪一棵树。
- **`acc_reusable_as_input()`** 回答的是：「一个矩阵累加器能否直接回喂、当作下一个乘法的操作数？」在 CDNA（MFMA 累加器与输入共享 `RT_16X16`）、gfx12（每个角色同一个片段）、CUDA（两半式 f32 累加器所持有的 `m16n8` C 片段恰好处于 A 操作数的寄存器次序）以及 Metal（B 与累加器共享同一个映射）上为真。仅在 gfx11 上为假：偶/奇的 `<8×f32>` 累加器与复制式的 `<16×in>` 操作数并不相同，于是这个值得经 LDS 往返一趟重新布局。[Flash Attention](./flash-attention) 在它的两个 matmul 之间按此分支。

映射针对一个 `LaneArith` trait 只写一次，却被求值两次：构建内核时在 `Index` 类型的 UOp 上求值，在 `tk/src/test/unit/layout.rs` 中则在普通整数上求值，那里每个变体都被证明是双射，Apple 的映射还被钉死在一张于硬件上实测得出的表上。`LaneMap::ldmatrix_x4` 从同一个闭式推导出 CUDA `ldmatrix` 的寄存器方案，而不是手工排列。`BaseShape` 上的 `ept` 字段（来自 [什么是分块](./tiling)）也出于同样的理由而存在：gfx11 上操作数被跨 lane 复制，所以每线程元素数并不等于 `element_count / wave_size`，必须显式存储。
:::

---

## 为什么这很重要

跨 wave 大小与片段布局的可移植性，是手写内核身上要付的那笔税，也是为什么把一个 NVIDIA tile 库朴素移植到 AMD 上根本跑不通。`tk` 把这笔税一次性付清，付在 `ArchCaps` 与 `LaneMap` 这两个抽象里，于是各个内核保持可读：它们只用*角色*说话，把那些 lane 的事交给硬件表去摆平。而同一层抽象随后还能把一个内核带*上* NVIDIA 与 Apple——那里 warp 永远是 32，片段布局却是各自核心的一套——这正是当初把它建起来所换来的回报。[Flash Attention](./flash-attention) 就是你看到这套做法在一个真实内核里得到回报的地方。
