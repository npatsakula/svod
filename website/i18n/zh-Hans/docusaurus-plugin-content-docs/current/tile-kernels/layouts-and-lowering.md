---
sidebar_label: 布局与降级
---

# 布局与降级

`lower::lower(program, &lowering, params, device)` 把 tile 程序变成一个指令列表已排好顺序的 Svod 程序。`Lowering` 包含所有作者没有决定的东西：`Device`（目标及其代价）、`Schedule`、块 tile 上的 `WarpGrid`，以及共享内存行是否做 swizzle。每种内核配置都会构造自己的 `Lowering`（`GemmCfg::lowering`、`FaCfg::lowering`、`NormCfg::lowering`）。

| 步骤 | 模块 | 作用 |
|---|---|---|
| 1. 操作数物化 | `lower` | 位于共享内存或全局内存的 `mma` 操作数获得一次显式加载，载入新的寄存器 tile |
| 2. 屏障 | `lower::sync` | 对共享内存效果进行同步：插入屏障，并在使用异步拷贝的 tile 之前插入一次提交和等待。此时 `Pipeline` 仍是一条语句，其模板自行同步自己的环形缓冲，因此该 pass 只在环形缓冲与周围代码相接之处同步 |
| 3. 调度展开 | `schedule::expand` | 每个 `Pipeline` 变为循环、分支和显式的 `Sync` 语句 |
| 4. 布局推断 | `layouts::infer` | 每个寄存器 tile 获得一个布局；冲突处插入 `Relayout` |
| 5. 发射 | `lower/emit.rs` | 语句按程序顺序发射到 `UOp::linear_program` 中 |

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

这套代数提供 `compose`、`inverse`/`pseudo_inverse`、`product`（把一个布局平铺到另一个布局之上）、`transpose`、`sublayout` 和 `slice`。`test/unit/layout.rs` 中的测试把每个原子与其编码的闭式公式做穷举对比。

寄存器 tile 的布局是 `layouts::TileLayout { frag, reps, warps }`。它包含一个定义在 `(Reg, Lane)` 上的片段，在每个 warp 的子 tile 内重复 `reps` 次，以及各 warp 的子 tile 坐标。

## 原子 {#atoms}

`atoms::MmaAtom` 是一条矩阵核心指令，带有其 `a`、`b`、`c` 操作数的布局。`Target::mma(dtype_in, dtype_out)` 负责查找它。内核从不指名原子：推断从消费这些值的原子那里读取操作数布局。

| 目标 | 原子 (bf16/f16 → f32) | 片段布局 |
|---|---|---|
| CUDA | `mma.sync` m16n8k16 | `mma_sync_a`, `mma_sync_b`, `mma_sync_c` |
| CUDA sm_90 | `wgmma` m64nNk16，四个 warp 一起 | `wgmma_a`、`wgmma_c`；`B` 经描述符读取，从不驻留寄存器 |
| AMD CDNA | MFMA 16×16×16 | `mfma_16x16x16` |
| AMD RDNA3 / RDNA4 | WMMA 16×16×16 | `wmma_gfx11_*` / `wmma_gfx12` |
| Apple | simdgroup 8×8×8 | `simdgroup_8x8` |

`atoms::GroupMmaAtom` 是一条 warpgroup 指令：四个相邻的 warp 一起发出一个 64 × n 的乘积，操作数通过描述符从共享内存读取（`B` 总是如此，`A` 除非是寄存器 tile），累加器分布在整个 group 上，每个 warp 以 `mma.sync` 的 C 布局持有自己的 16 行。`core` 为 `Group` 的 GEMM 配置把乘积降低到它上面（`Lowering::group`）：网格的行是 warpgroup，共享操作数留在原地而不被加载，发射器把每个乘积写成一条跨 ring 的 swizzle 行的 `wgmma` 链（描述符的 swizzle 模式就是降低器自己的 XOR，因此 ring 的槽位从 1024 字节边界开始）。Blackwell 的 `tcgen05.mma` 是同一类，只是累加器位于 tensor memory、由单个线程发出（`AccResidence`、`IssueScope`）。

目前 CUDA 和 AMD 两行都会规划内核并在硬件上运行（已测量 sm_86 和 gfx1201；参见[可移植性](./portability)）。

## 布局推断 {#layout-inference}

1. **原子提供初始布局。** `mma` 把原子的布局按 `WarpGrid` 平铺后赋给其操作数和结果（`MmaAtom::issue`），并按在降级的 `Costs` 下使程序重布局代价最小的方向（`ir::Orient`）发射：`Cᵀ = Bᵀ·Aᵀ` 是同一条指令、只是交换了槽位，因此在方形原子上，累加器可以零代价地作为下一次乘积的操作数保留。作者可以固定方向（`Kernel::mma_oriented`）。
2. **逐元素操作统一布局。** `binary`、`cast`、`where_`、`mul_add` 的操作数和结果，以及循环的携带值，共享同一布局。
3. **无约束的 tile 取自然布局。** 到达不动点时，没有任何约束的 tile 获得 `layouts::natural`：每个 lane 持有最多 8 个元素的短行向量，lane 先沿列再沿行遍历，warp 按行切分。tile 先于向量获得布局。
4. **向量跟随其 tile。** `[rows, 1]` 或 `[1, cols]` 向量（例如 `reduce` 的输出或偏置行）被要求采用其 tile 所蕴含的行或列布局。推断不会脱离该 tile 为它单独设定默认布局。
5. **冲突变为重布局。** 两个消费者需要不同布局的值，会被插入一个 `TileOp::Relayout`。`TileLayout::relayout` 把它归类为 `Identity`、`RegPermute`（在每个 lane 内）、`Exchange`（每个寄存器通过 lane-xor 移动从固定伙伴取得，在每个 warp 中相同）或 `ViaSmem`。`Exchange` 只在其工作量的代价低于一次经由共享内存的往返时保留（`Relayout::work × Costs::lane_move < Costs::smem_trip`，工作量按每个寄存器计，为其互不相同的非零掩码移动数加上每多取一处的一次选择，一次选择按一次移动计）；否则该 tile 走这次往返。发射器用目标的 lane 间移动（`atoms::LaneMove`：RDNA 上为 DPP 和 `permlanex16`，CDNA 的 wave64 上为 `ds_bpermute`）实现 `Exchange`，而 `ViaSmem` 经由共享内存实现。`Relayout::cost_per_element` 依据 `Costs` 为每一类计价（`RegPermute` 免费，因为编译器会重命名寄存器；交换的工作量按 `lane_move` 计；`smem_trip`），方向搜索还对 lane 分散的全局加载的每个元素收取 `Costs::scatter`，因此 RDNA3 上带 bias 的注意力保持乘积为直接方向，并沿行读取 bias。

## 调度模板 {#schedule-templates}

`schedule::Schedule::Uniform { prefetch, unroll }`：每个 warp 都负责加载和计算。

| `prefetch` | `pipeline(extent, stages, …)` 的展开方式 |
|---|---|
| `CpAsync`, `stages ≥ 2` | 一个 `extent + stages − 1` 次迭代的循环。每次迭代等待直到至多 `stages − 2` 个拷贝组未完成，然后执行屏障并消费第 `i − (stages − 1)` 步。接着把第 `i` 步的 `cp.async` 拷贝发往上一次迭代释放的槽，并把它们作为一组提交（超出末尾时提交空组，使组数保持一致）。 |
| `CpAsync`, `stages = 1` | 拷贝、提交、等待全部完成、屏障、消费、屏障。 |
| `RegisterStaged`（1 级） | 同样的三个循环，只用一个槽：第 `i` 步的全局 → 寄存器加载，消费第 `i − 1` 步，屏障（所有 warp 用完该槽），寄存器 → 共享存储，再一次屏障；同样的共享内存可容纳深一倍的 tile。 |
| `PingPong`（`crews: 2`，1 级） | 两组 warp（warp 网格的两行）在同一个槽上以相差一个簇的节奏运行同一条簇流：一次行程按乘积切成加载簇（先是下一步的全局加载，同一操作数的全部加载在同一簇中，使其一行的各 k 切片共享同一缓存行；然后是该乘积的片段读取；最后一簇中按加载顺序提交）和提高发射优先级的乘积簇，每簇之后一个块屏障，于是一组从寄存器计算乘积时另一组在加载。头部行程填充第 0 步；第二组在循环前多等一个屏障，第一组在循环后多等一个。CDNA。 |
| `RegisterStaged`（2 级） | 第 `i` 步的全局 → 寄存器加载，消费第 `i − 1` 步，寄存器 → 共享存储，屏障。 |

`unroll` 为每个环形槽复制一份循环体，使槽地址运算折叠为常量。它是否有益是按配置测得的，而非假定（参见[内核库](./kernel-library#gemm)中的 GEMM 表）。

异步拷贝和分级拷贝归模板管理，由模板负责同步。`lower::sync` 处理其余情况：由同步拷贝写入的共享 tile 在被其他线程读取之前加屏障，自上次屏障以来被读过的 tile 在被覆盖之前加屏障。

## 发射 {#emission}

发射器在预线性化程序中按程序顺序列出每条语句的指令，因此 Svod 线性化器的拓扑排序从不在 tk3 内核上运行。超出视图边界的全局访问会被门控：加载读为零，存储被丢弃。当每行的块数为 2 的幂时，共享内存行按 16 字节块做 XOR swizzle。在布局允许时，16 位 tile 从共享内存到寄存器的加载使用 `ldmatrix.x4`。`Scale`（恰为乘积）把乘法放在一个分支之下，判断线程持有的任一因子是否不为 1，所有因子都为 1 的 wave 会跳过它。向零方向转换为 bf16 保留 f32 位的高半部分。bf16 tile 到 f32 的归约在 f32 中折叠：目标具备 `v_dot2_f32_bf16` 时（RDNA3 及以后、gfx950），lane 的各对按所保存的值相加，每对一条指令；否则每个元素先扩展。在 AMD 上，f32 的 `Exp2` 就是硬件的 `v_exp_f32`（`llvm.amdgcn.exp2`），没有 `llvm.exp2` 为保持 2⁻¹²⁶ 以下结果为非规格化数而加的范围归约；那些结果会刷为零。CUDA 后端把超出 48 KB 静态上限的共享内存放在一个动态数组中。配置候选会按目标的 opt-in 上限（`Target::smem_bytes`）过滤。

:::note[由发射器而非作者维护的不变量]
纯值被列在其输入所需的循环层级。寄存器访问是标量访问。共享与全局访问是 `SHRINK` 向量访问。门控存储位于 `If` 内部。携带值在定义它们的块末尾移入其 `phi`。这些规则在 `lower/emit.rs` 中强制执行，内核作者从不需要处理它们。
:::

对已记录但尚未降级的内容返回 `lower::Error::Unsupported`：`raw` 语句和寄存器转置。没有 `cp.async` 的目标上，异步拷贝就是同步拷贝，其 commit 与 wait 不发射任何指令。`TK3_DUMP_LIST=1` 会打印发射出的列表（参见[测试与调试](./testing-and-debugging)）。
