---
sidebar_label: tk、HipKittens 与 CuTile 对比
---

# 编写 tile 内核的三种方式

`tk` 并没有发明 tile 抽象。它身处一个小小的、基于 tile 的内核系统家族里；而理解 `tk` 设计最有用的办法，就是把它摆在两个血缘最近的亲戚旁边：

- **[HipKittens](https://github.com/HazyResearch/HipKittens)**：HazyResearch 面向 AMD 矩阵核心的 C++ tile 库，是 `tk` 各抽象的直系血脉。
- **[CuTile](https://github.com/NVIDIA/cutile-rs)**（cutile-rs）：NVIDIA Research 面向 NVIDIA GPU 上 tile 内核的 Rust 系统。

三者都共享 [什么是分块](./tiling) 里那个核心理念：把片段大小的 tile 拉进寄存器，对它们计算，再写回去。它们的分野在于*谁掌控硬件映射*，而这种分野是一道光谱。

---

## 这道光谱：显式控制 ↔ 托管抽象

这三个系统落在同一根轴上。一端是你亲手管理寄存器、共享内存和指令调度；另一端是你写 tile 级的代码，由下游编译器决定它如何映射到线程、共享内存和矩阵指令。HipKittens 落在显式那一端，CuTile 落在托管那一端。`tk` 则偏中线靠左：它像 HipKittens 一样给你显式的寄存器 tile 和共享 tile，但它不是一个独立后端，而是降级进 Svod 那唯一的一套 UOp IR。

---

## 并排对比

| 轴 | **tk** | **HipKittens** | **CuTile** |
|------|--------|----------------|------------|
| **编写界面** | Rust *构建器 API*（`Kernel`/`Group` 铸造 UOp） | C++ *模板* | Rust *宏 DSL*，在 `#[cutile::module]` 里写普通 Rust，由宏捕获 AST |
| **IR 目标** | Svod 的**唯一 UOp IR**，与整个编译器一致 | 无（模板 → clang amdgcn） | 一个*单独*的 MLIR `cuda_tile` 方言，序列化为 Tile IR 字节码 |
| **降级** | Svod render → LLVM → AMD 二进制，或 → PTX（由 `ptxas` 汇编成 cubin，否则由驱动 JIT），或 → MSL | clang | 字节码 → 外部 `tileiras` 汇编器 → cubin（首次启动时 JIT） |
| **内存模型** | **显式**的寄存器*和*共享 tile | 显式的寄存器*和*共享 tile | **一种** tile 类型（寄存器驻留）；共享内存分阶段是隐式的，由编译器选择 |
| **矩阵核心 API** | 显式的 `WMMA` 操作 + 基于角色的片段 | 带类型的 tile → `__builtin_amdgcn_mfma_*` | 单个函数式的 `mma()` 内建函数 |
| **计算/内存重叠** | 一个 `sched::pipeline` 标记 + 一个 codegen 遍 | 逐内核手写（原始调度内建函数） | 委托给 `tileiras` |
| **核心差异** | 一套 IR ⇒ 手写内核与自动调优内核平级 | 「从硬件向上构建」 | 跨越启动边界的内存安全 |
| **目标** | AMD CDNA3 / RDNA3 / RDNA4、NVIDIA `sm_80+`、Apple7+ | AMD CDNA / RDNA | 仅 NVIDIA `sm_80+` |

在此之上，每个 `tk` 内核还各自声明自己的架构集合：flash attention 和方阵 `matmul` 在所有家族上运行，`gemm_nt` 和各个 norm 在 RDNA 与 CUDA 上运行，单查询注意力在 AMD 与 CUDA 上运行，k-means 与 k-NN 仅限 AMD——完整矩阵见 [内核库](./kernel-library)。

---

## 代码长什么样

这几种编写界面的使用体验确实大不相同。HipKittens 和 CuTile 的片段只作示意，传达的是各模型的*形态*，并非精确的 API；`tk` 的片段则是真实的 API。

**HipKittens**：C++ 模板；你给 tile 命名，直接调用乘法：

```cpp
using namespace kittens;
rt_bf<64, 32>      a, b;     // register tiles of bf16
rt_fl<64, 32, col> acc;      // fp32 accumulator, col layout (MFMA output)

load(a, a_global, {row, k});
load(b, b_global, {k, col});
mma_ABt(acc, a, b, acc);     // acc += a · bᵀ  → __builtin_amdgcn_mfma_*
```

**CuTile**：在一个由宏捕获的模块里写普通 Rust；tile 不可变，共享内存由编译器替你分阶段：

```rust
#[cutile::module]
mod kernels {
    use cutile::core::*;
    pub fn gemm(a: &Tensor<f32, A>, b: &Tensor<f32, B>, c: &mut Tensor<f32, C>) {
        let (i, j) = (tile_block_id_x(), tile_block_id_y());
        let mut acc = Tile::<f32, ACC>::zeros();
        for k in 0..a.dim(1) / BK {
            acc = mma(a.partition(AK).load([i, k]),
                      b.partition(BK).load([k, j]),
                      acc);            // one functional intrinsic
        }
        c.partition(CC).store([i, j], acc);
    }
}
```

**tk**：一个铸造 IR 的 Rust 构建器；你按角色请求片段，再发射 `Group` 操作（取自 `tk/src/kernels/gemm.rs` 中 `gemm_core` 的单缓冲条带，有所精简）：

```rust
let (outs, ins) = ker.bind_abi(&[GlSpec::new(&[1, 1, n, n], DType::Float32)], &[a_spec, b_spec]);
let g = ker.group_2d(cfg.warps_m, cfg.warps_n);
let a_smem = ker.shared_sw((cfg.block_m, k_step), bf16, TileLayout::Row);   // swizzled LDS strip
let acc = g.zero(ker.acc((reg_m, reg_n), TileLayout::Col));                 // role, not a hardcoded shape

let lp = ker.loop_static(trips);
let a_f = g.fill_local_nobar(a_smem, a_gl, &a_idx, 2);                      // global → LDS, collaborative
let a_sub = g.load(ker.operand((reg_m, k_step), bf16, TileLayout::Row),    // LDS → registers
                   a_f.subtile((reg_m, k_step), (warp_row, 0)), MoveIdx::default());
let acc = g.mma_ab(acc, &a_sub, &bb);                                      // → WMMA UOp
let ended = lp.close();
let _ = g.store(c_gl, acc.after(&ended), MoveIdx::block((0, 0, mrow, nidx), 2));
ker.finish(1)                                                               // SINK { opts_to_apply: Some(vec![]) }
```

CuTile 的例子读起来像一段普通程序，`tk` 的例子读起来像在搭一张图。这就是其中的权衡：CuTile 的宏捕获你的*语法*再重新解析，而 `tk` 是一个库，它的方法调用*本身*就是 IR 构造。

---

## 关键的概念差异

有两点比其余的都更要紧。

**谁拥有共享内存。** CuTile 恰恰只有*一个* tile 概念，即寄存器 tile，并刻意把共享内存分阶段藏了起来；它的 `tileiras` 汇编器自行决定数据如何流经 LDS、缓存和矩阵核心。`tk` 和 HipKittens 则把寄存器 tile *和*共享 tile *两者都*亮出来，要你显式地分阶段。CuTile 站在寄存器/共享之分*之上*一层，`tk` 则正站*在那一层上*。这便是掌控带来的代价与威力：要管的东西更多，但那些赢得性能的[重叠与 swizzle 决策](./where-flops-hide)，都由你来拍板。

**IR 驻留在哪里。** 这才是 `tk` 真正与众不同的一招。HipKittens 是一个独立的 C++ 框架，它产出内核，仅此而已。CuTile 降级到一个*单独*的 MLIR 方言，只有它自己的工具链才消费它。`tk` 降级进的，是 **Svod 其余部分早已通晓的那同一套 UOp IR**。一个 `tk` 内核不是一件交给另一个编译器的产物，而是那唯一 IR 中的一个子图，就紧挨着每一个自动调优内核。

:::tip[面向 GPU 专家]
IR 目标的差异，在工具链层面是实打实的。`tk` 把它的 `SINK` 经 `svod-codegen` 渲染到 LLVM IR，再到一个 AMD 二进制、或到 PTX（由 `ptxas` 汇编，否则由驱动 JIT），在 Metal 上则到 MSL，走的与图内核是同一条路。CuTile 则把它的 tile 方言序列化为字节码，由一个*外部*的 `tileiras` 汇编器变成 cubin，并在首次启动时 JIT 编译；HipKittens 则是由 clang 编译的 C++ 模板。所以 `tk` 的「一套 IR」，字面上就意味着一条渲染加编译的流水线，而其他两者都要桥接进一个单独的编译器。
:::

---

## 为什么这很重要

正是这一点，让 Svod 能同时端出两样通常互斥的东西：既让编译器去找调度，又让你自己写调度，而且不必搭第二个编译器。

一个 BEAM 自动调优的 matmul 和一个手写的 Flash Attention，都不过是同一张 DAG 里的 `SINK` UOp。它们经同一个渲染器渲染，在同一个运行时上运行，用同一个调试器打印。唯一把它们区分开的，是那个 `opts_to_apply` 标记，它的归宿在 [向 IR 中编写](./lowering)：同一套 IR，既承载优化器驱动的内核，也承载手工驱动的内核。

HipKittens 证明了从硬件向上能够匹敌厂商库，CuTile 证明了 GPU 内核可以既安全又高层。`tk` 的押注更聚焦，对 Svod 也更有用：把那套从硬件向上的 tile 模型拿过来，但不去围着它另起一个后端，而是*直接说编译器早已拥有的那套 IR*。这就是 `tk` 之所以小巧的全部原因，也是手写内核为什么感觉像一等公民、而不是一个逃生舱口的原因。
