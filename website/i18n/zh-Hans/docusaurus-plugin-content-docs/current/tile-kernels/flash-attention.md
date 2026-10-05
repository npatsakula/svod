---
sidebar_label: Flash Attention
---

# 实战范例：Flash Attention

Flash Attention 正是那个为 `tk` 的存在撑起理由的内核，也就是 [概览](./overview) 点名*无法*表达成单一可调度规约的那个，是手工编写面孔之所以存在的全部缘由。本章带你走一遍：它难在哪里，tile 抽象如何应对，以及 [布局分歧](./wave-portability) 在何处实打实地显现。

我们要讲的是 `tk/src/kernels/fa.rs` 里的前向内核（`build_fa_mw_rdb`），从 USE 面孔的 `flash_attention(q, k, v)` 和 `flash_attention_with(q, k, v, opts)` 进入。它为 [目标架构表](./kernel-library#targets) 中的每个家族构建：CDNA3、RDNA3.5、RDNA4、CUDA `sm_80+` 和 Apple7+。

---

## 为什么 attention 无法被自动调优

朴素的 attention 是 `softmax(QKᵀ) · V`。直白地写出来，意思就是：构造完整的 `N×N` 得分矩阵，对它做 softmax，再乘以 `V`。这个得分矩阵大得吓人，而且从来不必一次性全部存在，于是 Flash Attention 流式扫过一块块 key 和 value，*增量地*维护 softmax。

「增量」这个词正是症结。softmax 的归一化依赖于*所有* key 上的最大值与求和，可我们一次只看到一块。于是我们维护一份当前的统计量，边推进边修正结果。这就是**在线 softmax**，而它是一个递推：每一个 KV 块都要读取并更新上一块产出的状态。

优化器能做的动作只是「把这个 `REDUCE` 分块、展开」。可这里根本没有 `REDUCE` 供它分块，有的是一个循环，循环体依赖自己上一轮的迭代。搜索找不出它，你只能亲手写。

---

## 用 tile 表达的算法

一个工作组由八个 wave（`NUM_WARPS`）组成，负责一个 `(head, q-block, batch)` 三元组——启动网格为 `[H, N / (q_blk · 8), B]`。每个 wave 拥有一块 `q_blk × D` 的查询 tile，在整个内核期间驻留在寄存器中；八个 wave 共享共享内存中的同一个 K/V 块，并协作填充它。对每个含 `kv_blk` 个 key 的 KV 块，wave 跑下面这段循环体，全程都是 tile 操作：

```text
for each block of K, V:                          ┌─ everything here is a tile op
    S   = Q · Kᵀ                                 │  mma_atb into a zeroed f32 accumulator
    S   = S · log2(e)/√D                         │  the softmax scale, on the f32 scores
    S   = mask(S)                                │  causal + key-padding + segment masks
    m'  = max(m, colmax(S))                      │  update running max  (cross-lane reduce)
    P   = exp2(S - m')                           │  rescale to the new max (base-2 exp)
    l   = l · exp2(m - m') + colsum(P)           │  update running sum
    O   = O · exp2(m - m') + P · V               │  rescale accumulator, accumulate (mma_atb)
    m   = m'                                     │
O = O / l                                        └─ final normalize, transpose, store
```

每块两次矩阵乘法（`Q·Kᵀ` 和 `P·V`）、两次跨 lane 规约（最大值和求和），以及每当当前最大值变动时对输出累加器的一次重新缩放。那个 `exp2`（以 2 为底的指数）是刻意为之，这样就能直接用上硬件的快速 `exp2` 单元。缩放因子 `log2(e)/√D` 作用在 f32 得分累加器上，而不是预先折进 `Q`：缩放 `Q` 会让它再被舍入一次到 16 位操作数 dtype，这个误差按得分自身的量级进入得分，随后又被 `exp2` 放大。

那几行里的每一行，都是对 tile 的一个 `Group` 操作（`tk/src/kernels/fa.rs` 中的 `fa_qk` 和 `fa_softmax_pv`）。得分 tile 是列主序的 `(KV, Q)`，因此 softmax 用 `col_reduce` 沿它的*高度*规约成每个查询一份的 `RV`，而重新缩放用的则是 [构建器 API](./builder-reference) 中的运算符语法糖：

```rust
let max_vec_last = warp.copy(lp.reinit(max_vec_last), &max_vec);
max_vec = warp.col_reduce(max_vec.after(&max_vec_last), &att, |a, b| a.max(b), f64::NEG_INFINITY);
let scale_vec = (max_vec_last - &max_vec).exp2();
o_reg = o_reg * &scale_vec;
norm_vec = norm_vec * &scale_vec;
let att = (att - &max_vec).exp2();
norm_vec = warp.col_reduce(norm_vec.after(&scale_vec), &att, |a, b| a.add(b), 0.0);
```

全程看不到一点 lane 算术。递推还逼出了一个细节：当前最大值从 `f32::MIN` 而非 `−∞` 起步，这样当某个块被掩码对某个查询行完全遮住时，得到的是 `exp2(m − m') = 1`，而不是 `−∞ − (−∞) = NaN`。

---

## 流式传输：双缓冲的 KV

这是 [FLOPS 藏在哪里](./where-flops-hide) 里瓶颈 2 的实战。当矩阵核心在处理当前 KV 块时，下一块就该已经在通往共享内存的路上了。内核为每个操作数维持**两个** LDS 半区（`ker.shared_db`），并按循环计数器的奇偶交替使用：在半区 `kv % 2` 上计算的同时加载半区 `(kv + 1) % 2`。

```text
   load K/V block 0 --> LDS[A]
   ┌─────────────────────────────────────────────────┐
   │ compute on LDS[A]   ║   load block 1 --> LDS[B] │   <- overlap
   │ compute on LDS[B]   ║   load block 2 --> LDS[A] │
   │ ...                                             │
   └─────────────────────────────────────────────────┘
```

下一块如何传输由架构决定，具体由 `Group::cp_async_fill_applies` 判定：

- **CUDA sm_80+**：`cp.async` 把全局内存直接拷到共享内存，不经寄存器中转。循环顶部先收尾上一次发出的拷贝（`cp_async_wait(0)` + 屏障），把块 `kv+1` 发往另一个半区，然后在拷贝进行中对当前半区做收集——每个 16 位片段一条 `ldmatrix.x4`。
- **AMD 和 Metal**：寄存器中转的流式传输。`stage_global_to_reg` 在 MMA 之前把块 `kv+1` 的全局加载发进每个 lane 的寄存器，`commit_regs_to_local` 在 MMA 之后把它们写进另一个半区；每次迭代一个 `war_fence2` 屏障——由收集操作消费，并以提交作为依赖——同时覆盖了刚写入的半区上的写后读，以及每个 wave 刚收集完的半区上的读后写。

预取索引按块数取模回绕，因此最后一次迭代会重新读取块 0（它从不被收集），而不会越出操作数的边界。KV 循环本身是一个带*动态*上界的 `Loop`：在 `causal` 下，每个 q-block 只访问 `(block_q_base + 1) · 8 · q_blk / kv_blk` 个超块，这就是因果块跳过。

---

## 布局的细节：两个 matmul 之间的重新布局

这里就是 [布局与 wave 宽度](./wave-portability) 不再抽象的地方。内核做两次矩阵乘法，第一次的输出（`S = Q·Kᵀ`，经 softmax 后变成 `P`）是第二次（`P·V`）的*输入*。得分累加器能不能直接回喂、当作操作数？

- **在 CDNA、RDNA4、CUDA 和 Metal 上**（`acc_reusable_as_input() == true`）：能。累加器与操作数共用一张 lane 映射——CDNA 上的 MFMA 片段、gfx12 上唯一的那种片段、CUDA 上按 A 操作数寄存器次序排列的 `m16n8` C 片段、Apple 上唯一的 `simdgroup_matrix` 映射——所以 `att_mma` 只是一次带 f32 → 16 位转换的寄存器 `copy`。
- **在 RDNA3 上**（`acc_reusable_as_input() == false`）：不能。偶/奇累加器与复制式操作数并不相同，所以 `P` 要**经 LDS 往返一趟**：先用 `store_local_fenced` 把累加器存进一块每工作组 `[8 · kv_blk, q_blk]` 暂存 tile（`att_smem`）中属于本 wave 的那条带，再按操作数映射 `load` 回来。`FaPolicy::att_band` 会报告这一点，好让共享内存预算把这条带计算在内。

内核只在分配 `att_smem` 时基于 `ker.caps.acc_reusable_as_input()` 分支一次；热循环读取的是 `Option<ST>`。同一个算法，两种物理实现，正是上一章所说的那笔可移植性税，落在最重要内核的最热循环里。

---

## 掩码

因果掩码（一个查询不能关注一个未来的 key）、key 填充掩码（`key_lens[b]`，忽略位于有效数量及其之后的 key）和打包序列分段（`seg_start[b, q]`，忽略查询自身分段之前的 key）都在 softmax 之前作用到得分 tile `S` 上。三者都是 `mask_where` 调用：每个得分元素的位置，由「是哪个片段、哪个 lane 持有它」隐含给出，所以掩码是从 tile 自身的 `LaneMap` 算出来的，不是取出来的。可选的掩码张量是排在末尾的全局缓冲区，在 `o, q, k, v` 之后用普通的 `ker.gl` 调用绑定，因此不带掩码时的 ABI 保持不变。

---

## 每 warp 的 tile

`FaConfig { q_blk, kv_blk, unroll, causal }` 是循环体的调优旋钮。基线是 `{16, 32}`——16 行的 Q tile 加 32 个 key 的 KV 超块，与 `{16, 16}` 相比，它提高了每个 wave 的 MMA 指令级并行，并把 softmax 的簿记工作减半。`FaPolicy`（`tk/src/kernels/fa.rs`）在启动网格覆盖设备计算单元、且头维度低于该家族上限时，为每个家族选择更高的 tile；若某个头维度的缓冲区会超出共享内存（AMD 上 64 KiB，CUDA 上静态 48 KiB，Apple 上 32 KiB），则拒绝。首次启动时会对 `FA_TILES` 的四个候选计时并缓存胜者——见 [自动调优](./tuning)。

`unroll` 只在 CUDA 上开启：NVPTX 后端只有在每个寄存器索引都是常量时才会把累加器留在寄存器中，所以循环体被平铺发射（`Kernel::set_unroll`）；AMD 保留折叠形式。

:::tip[面向 GPU 专家]
计算/内存的重叠在 `tk` 里并不像 HipKittens 的内核那样被手工发射成原始调度内建函数。这里的 KV 循环被标注上 `sched::pipeline(SchedKind::Attention, kv_idx)`（`tk/src/kernels/fa.rs`），这是一个穿过循环内 K/V 缓冲区的标记，由 `codegen/src/llvm/sched.rs` 中线性化之后的遍来消费。内核体表达的是*要重叠什么*，指令排序则交给这个遍决定。目前它在 CDNA 上把标记降级为 `@llvm.amdgcn.iglp.opt(0)`——后端预置的 MFMA/内存交织——在其他平台上则只留下一条无作用的注释；计划中，attention 这一类将换成一种感知 softmax 的梳状编排，把指数运算织进矩阵运算的间隙。
:::

---

## 为什么这很重要

Flash Attention 是整节内容浓缩进一个文件：

- 它之所以存在，是因为**在线 softmax 是一个递推**，而非一个可分块的规约（[概览](./overview)）；
- 它的成败系于**流式传输与重叠**（[FLOPS 藏在哪里](./where-flops-hide)）；
- 它完全用 **tile 和角色**表达，从不用 lane 索引（[什么是分块](./tiling)）；
- 它编译成与其他一切**相同的 UOp IR**，并作为一个 `Op::Call` 融入惰性图（[向 IR 中编写](./lowering)）；
- 而且它的热循环里携带一个显式的**累加器复用分支**，每种片段布局各一条（[布局与 wave 宽度](./wave-portability)）。

这就是它为什么要手写，以及 `tk` 为什么为写它而存在。要隔离运行它、检验它的数字，见 [调试](./debugging)。
