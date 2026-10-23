---
sidebar_label: 布局与降级
---

# 布局与降级 {#layouts-and-lowering}

`lower::lower(program, &lowering, params, device)` 把 tile 程序变成一个指令列表已排好顺序的 Svod 程序。`Lowering` 包含所有作者没有决定的东西：`Device`（目标及其代价）、`Schedule`、块 tile 上的 `WarpGrid`，以及乘积在哪种矩阵指令上发出（`core`）；共享内存行总是做 XOR swizzle。每种内核配置都会构造自己的降级参数（`GemmCfg::lowering`、`FaCfg::lowering`、`NormCfg::lowering`）。`lower::standalone(program, &lowering)` 针对未连接设备的目标进行降级，`lower::render` 返回将交给后端编译器的 LLVM 文本。

| 步骤 | 模块 | 作用 |
|---|---|---|
| 1. 操作数物化 | `lower` | 位于共享内存或全局内存的 `mma` 操作数获得一次显式加载，载入新的寄存器 tile；在 warpgroup 核心上，共享操作数留在原地 |
| 2. 屏障 | `lower::sync` | 对共享内存效果进行同步：插入屏障，并在使用异步拷贝的 tile 之前插入一次提交和等待。此时 `Pipeline` 仍是一条语句，其模板自行同步自己的环形缓冲，因此该 pass 只在环形缓冲与周围代码相接之处同步 |
| 3. 调度展开 | `schedule::expand` | 每个 `Pipeline` 变为循环、分支和显式的 `Sync` 语句 |
| 4. 布局推断 | `layouts::infer` | 每个寄存器 tile 获得一个布局；冲突处插入 `Relayout` |
| 5. 共享内存方向 | `lower` | 经由寄存器填充的共享分配，沿其读者片段延伸的轴存储，使收集操作成为 16 字节加载；`cp.async` 的 tile 保持行主序 |
| 6. 发射 | `lower/emit/` | 语句按程序顺序发射到 `UOp::linear_program` 中 |

当程序的 warp 数不等于 warp 网格的 warp 数（或在主体中有 `Warps` 语句时不是其倍数）时，`lower` 以 `Error::Grid` 失败；当块屏障位于按 warp 分支的分支之下、会被部分 warp 跳过时，以 `Error::DivergentBarrier` 失败。

## F2 线性布局 {#f2-linear-layouts}

`layout::Layout` 是 GF(2) 上的线性映射，把输入维度的比特映射到输出维度的比特（即 Triton 的“Linear Layouts”表述）。维度为 `Dim::{Reg, Lane, Warp, Block, Row, Col}`。映射为每个输入比特存储一个基向量。每种厂商片段、每个 XOR swizzle 和每个 `ldmatrix` 方案都是这样的映射。下面是 `layout/atoms.rs` 中 PTX `mma.m16n8k16` 的累加器：

```rust
/// PTX `mma.m16n8k16` C/D fragment (16×8, M×N): `row = g + 8·(c/2), col = 2t + c%2`.
pub fn mma_sync_c() -> Layout {
    Layout::from_bases(
        [(Row, 16), (Col, 8)],
        &[(Reg, &[[0, 1], [8, 0]]), (Lane, &[[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]])],
    )
}
```

这套代数提供 `identity`、`zeros`、`from_bases`、`transpose`、`product`（把一个布局平铺到另一个布局之上）、`sublayout` 和 `apply`。测试把每个原子与厂商 ISA 文档中的 lane 公式（`test/unit/atoms.rs`）以及 tk1 的闭式公式（`test/unit/layout.rs`）进行对比。

寄存器 tile 的布局是 `layouts::TileLayout { frag, reps, warps }`。它包含一个定义在 `(Reg, Lane)` 上的片段，在每个 warp 的子 tile 内重复 `reps` 次，以及各 warp 的子 tile 坐标。

## 原子 {#atoms}

`atoms::MmaAtom` 是一条矩阵核心指令，带有其 `a`、`b`、`c` 操作数的布局。`Target::mma(dtype_in, dtype_out)` 负责查找它。内核从不指名原子：推断从消费这些值的原子那里读取操作数布局。

| 目标 | 原子 (bf16/f16 → f32) | 片段布局 |
|---|---|---|
| CUDA | `mma.sync` m16n8k16 | `mma_sync_a`, `mma_sync_b`, `mma_sync_c` |
| CUDA sm_90 | `wgmma` m64nNk16，四个 warp 一起 | `wgmma_a`、`wgmma_c`；`B` 经描述符读取，从不驻留寄存器 |
| AMD CDNA | MFMA 16×16×16 | `mfma_16x16x16` |
| AMD RDNA3 / RDNA4 | WMMA 16×16×16 | `wmma_gfx11_input`、`wmma_gfx11_acc` / `wmma_gfx12` |
| Apple | simdgroup 8×8×8（仅原子；无内核） | `simdgroup_8x8` |

每个有内核的目标还有一个 8 位整数到 i32 的原子（`Target::geometry_of(Int8)`）：`mma.sync` m16n8k32 s8 和 CDNA 的 MFMA 16×16×32 i8 用 16 位片段的字节承载、深度加倍（`mma_sync_a8`、`mma_sync_b8`、`mfma_16x16x32_i8`）；RDNA 的 `iu8` WMMA 保留 16 位布局，深度为 16。它的 `m` 和 `n` 与 16 位原子相同。规划器按 16 位几何（`Target::geometry`）定价，8 位行上的问题除外。

`atoms::GroupMmaAtom` 是一条 warpgroup 指令：四个相邻的 warp 一起发出一个 64 × n 的乘积，操作数通过描述符从共享内存读取（`B` 总是如此，`A` 除非是寄存器 tile），累加器分布在整个 group 上，每个 warp 以 `mma.sync` 的 C 布局持有 16 行。`core` 为 `Core::Group` 的配置把乘积降级到它上面：网格的行是 warpgroup，共享操作数留在原地而不被加载，发射器把每个乘积写成一条跨环形缓冲 swizzle 行的 `wgmma` 链（描述符的 swizzle 模式就是降级自身的 XOR，即 `layouts::Swizzle`）。GEMM 和注意力的配置在 sm_90 上都提供这种核心。

## 布局推断 {#layout-inference}

1. **原子提供初始布局。** `mma` 把原子的布局按 `WarpGrid` 平铺后赋给其操作数和结果（`MmaAtom::issue`），并按在降级的 `Costs` 下使程序重布局代价最小的方向（`ir::Orient`）发射：`Cᵀ = Bᵀ·Aᵀ` 是同一条指令、只是交换了槽位，因此在方形原子上，累加器可以零代价地作为下一次乘积的操作数保留。结果会被分发的乘积（`TileOp::Deal`，列表的 lane 队列）按把一行摊到最少 lane 上的方向发射（`MmaAtom::narrow`）：在那里分发只是寄存器重命名，算子层也正是这样定价的。作者可以固定方向（`Kernel::mma_oriented`）。
2. **逐元素操作统一布局。** `binary`、`cast`、`where_`、`mul_add` 的操作数和结果，以及循环的携带值，共享同一布局。
3. **无约束的 tile 取自然布局。** 到达不动点时，没有任何约束的 tile 获得 `layouts::natural`：每个 lane 持有最多 8 个元素的短行向量，lane 先沿列再沿行遍历，warp 按行切分。
4. **向量跟随其 tile。** `[rows, 1]` 或 `[1, cols]` 向量（例如 `reduce` 的输出或偏置行）被要求采用其 tile 所蕴含的行或列布局。
5. **冲突变为重布局。** 两个消费者需要不同布局的值，会被插入一个 `TileOp::Relayout`。`TileLayout::relayout` 把它归类为 `Identity`、`RegPermute`（在每个 lane 内，免费）、`Exchange`（每个寄存器通过 lane-xor 移动从固定伙伴取得）或 `ViaSmem`。`Exchange` 只在其工作量的代价低于一次经由共享内存的往返时保留（`Relayout::work × Costs::lane_move < Costs::smem_trip`）。发射器用目标的 `atoms::LaneMove` 实现 `Exchange`（CUDA 上为 `shfl.sync`，RDNA 上为 DPP 和 `permlanex16`，CDNA 上为 `ds_bpermute`）。

## 调度模板 {#schedule-templates}

`schedule::Schedule::Uniform { fill, unroll, async_products }`：每个 warp 都负责加载和计算。`Schedule::PingPong { crew }`：两个各含 `crew` 个 warp 的工作组共用一个寄存器暂存槽（CDNA GEMM，`GemmCfg::crews == 2`）。

| 填充方式 | `pipeline(extent, stages, …)` 的展开方式 |
|---|---|
| `Fill::CpAsync`，`stages ≥ 2` | 一个 `extent + stages − 1` 次迭代的循环。每次迭代等待直到至多 `stages − 2` 个拷贝组未完成，然后执行屏障并消费第 `i − (stages − 1)` 步。接着把第 `i` 步的 `cp.async` 拷贝发往上一次迭代释放的槽，并把它们作为一组提交（超出末尾时提交空组，使组数保持一致）。 |
| `Fill::CpAsync`，`stages ≥ 3`，`async_products` | 与上相同，但消费阶段的 `wgmma` 链在下一次填充期间保持在途：被重新填充的是两次行程之前消费过的槽，每次消费之后，每个线程等待除最新一条之外的所有链完成（`Sync::WaitProducts`）。 |
| `Fill::CpAsync`，`stages = 1` | 拷贝、等待、消费，并在重新填充该槽之前插入一次屏障。 |
| `Fill::Staged`，2 级 | 在同一个归纳变量上的三个循环：第一次行程把第 0 步加载到寄存器并存入其槽；稳态行程把第 `i` 步加载到寄存器，同步，消费第 `i − 1` 步，同步，再把寄存器存入第 `i` 步的槽，其循环体中没有分支（在 RDNA 上这两道同步把加载留在乘积之前、把提交留在乘积之后）；最后一次行程只做消费。 |
| `Fill::Staged`，1 级 | 同样的三个循环，只用一个槽：存储在屏障之后等待所有 warp 用完该槽，然后再插入一次屏障，因此同样的共享内存可容纳深一倍的 tile。 |
| `PingPong` | 一次行程在其乘积处被切成加载簇和乘积簇，每簇之后一个块屏障，于是一个工作组以提高的发射优先级从寄存器计算乘积时，另一个工作组在加载。 |

`unroll` 为每个环形槽复制一份循环体，使槽地址运算折叠为常量；调优器对两种情况都做测量。异步拷贝和暂存拷贝归模板管理，由模板负责同步。`lower::sync` pass 处理其余情况：由同步拷贝写入的共享 tile 在被其他线程读取之前加屏障，自上次屏障以来被读过的 tile 在被覆盖之前加屏障。

## 发射 {#emission}

发射器在预线性化程序中按程序顺序列出每条语句的指令，因此 Svod 线性化器的拓扑排序从不在 tk 内核上运行。超出视图边界的全局访问会被门控：加载读为零，存储被丢弃。当每行的块数为 2 的幂时，共享内存行按 16 字节块做 XOR swizzle（`layouts::Swizzle`）。在布局允许时，16 位 tile 从共享内存到寄存器的加载使用 `ldmatrix.x4`，8 位 tile 则使用其非转置形式、覆盖 16×32 字节；在 RDNA4 上，当布局恰好是 `global_load_tr_b128` 所产生的布局时，从全局内存到寄存器的加载使用该指令。`Warps` 语句在每个 warp 上单独运行其主体，带有自己的拷贝和屏障。在 AMD 上，f32 的 `exp2` 就是硬件的 `v_exp_f32`，它会把低于 2⁻¹²⁶ 的结果刷为零。在 CUDA 上，超过 48 KB 的共享内存放在一个动态数组中；候选会按 `Target::smem_bytes` 过滤。

:::note[由发射器而非作者维护的不变量]
纯值被列在其输入所需的循环层级。寄存器访问是标量访问。共享与全局访问是 `SHRINK` 向量访问。门控存储位于 `If` 内部。携带值在定义它们的块末尾移入其 `phi`。
:::

`TK_DUMP_LIST=1` 会打印发射出的列表（参见[测试与调试](./testing-and-debugging)）。
