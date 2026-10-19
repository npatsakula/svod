---
sidebar_label: 内核库
---

# 内核库

`svod_tk3::kernels` 中的每个内核都是一个函数 `fn k<T: Elem>(spec: &Spec) -> Program`，对 16 位元素类型（`BF16` 或 `F16`）泛型。每个 spec 都有一个 `batch: Batch`：`Batch::Static(n)` 启动 `n` 个批次，`Batch::Var { name, min, max }` 按名称绑定的运行时变量的实际值启动批次，缓冲区按 `max` 分配。每种配置类型都有一个 `lowering(&device)`（一个 `atoms::Device`：目标及其代价）。模型通过[算子层](./op-layer)使用这些内核，由算子层选择配置。

| 内核 | Spec → 程序 | 内核名称 | 算子 |
|---|---|---|---|
| GEMM + 尾处理 | `GemmSpec` → `gemm::gemm` | `gemm` | `ops::linear` |
| 隐式 GEMM 卷积 | `ConvSpec` → `conv::conv`（拆分 K 时另加 `gemm::split_merge`） | `conv`、`split_merge` | `ops::conv2d` |
| Flash attention 前向 | `AttnSpec` → `attention::attention` | `flash_attention` | `ops::attention` |
| 拆分合并 | `CombineSpec` → `attention::combine` | `combine_splits` | 带 `splits` 的 `ops::attention` |
| 注意力前处理 | `HeadsSpec` → `heads::heads` | `heads` | `ops::heads` |
| LayerNorm / RMSNorm | `NormSpec` → `rows::norm` | `layer_norm` / `rms_norm` | `ops::{layer_norm, rms_norm, add_*}` |

## GEMM {#gemm}

`c = act(a·bᵀ + bias) + residual`，其中 `a [batch·m, k]`、`b [n, k]`。矩阵乘之后的所有计算都在 f32 累加器上进行，结果在存储时只舍入一次。

| `Epilogue` 字段 | 作用 |
|---|---|
| `bias` | 加到累加器上的 `[n]` 行（门控时为 `[2n]`） |
| `act` | `Act::None`、`Act::Gelu`（erf 近似，误差至多 1.5e-7）、`Act::Silu` |
| `gated` | `b` 为 `[2n, k]`，门控行在上、up 行在下；输出 `act(gate)·up`（SwiGLU、GeGLU） |
| `residual` | 最后加上的 `[batch·m, n]` |

`GemmCfg { tile: [bm, bn, bk], stages, warps: [rows, cols], group_m, unroll, core, crews }`。超出 `m` 的行和超出 `n` 的列是带边界的视图，因此 `m` 和 `n` 不受限制。`k` 必须是 `bk` 的倍数。

算子层不保留 tile 表。`config::Planner::gemm_ranked` 枚举该构造的各个旋钮（32–256 行、48–256 列的输出 tile，`bk` 16–64，九种 warp 网格，目标预取所运行的环形缓冲深度，展开与不展开），保留降级能够运行的配置（原子能铺满 warp 网格，填充能按每线程整 16 字节块拆分，环形缓冲放得进共享内存），再依据设备测得的 `Costs` 以 SM 时间的纳秒数为其余配置排序。一个块的一次行程沿三条管线发射：每个 wave 从共享内存读取的操作数片段、矩阵核步骤和流入的填充，外加行程的固定开销。这些管线在一个 SIMD 所容纳的 wave 之间重叠：最长的一条全额计入，其余按每 SIMD 的 wave 数分摊为一份，至多四个（`OVERLAP_WAVES`：在 gfx1201 和 sm_86 上，每 SIMD 十六个 wave 跑一个 tile 并不比四个更快）。一个 SIMD 上的 wave 少于 `step_lone / step`（隐藏矩阵核依赖链延迟所需的 wave 数）时，步骤就慢这么多倍，而一个 wave 的独立累加器按 wave 计数。最忙的 SM 发射 `per_sm = ⌈blocks / sms⌉` 个块的工作，一个块结束就接上下一个，行程的固定开销由它同时容纳的块分摊（`resident`，由共享内存和寄存器估计决定）；这些块的一轮至少是单个 wave 自身的发射（wave 数少于 SM 的 SIMD 数的块会让其余 SIMD 空闲），并且每一轮都在其发射之上再支付提前一级流水线无法隐藏的内存延迟：`cost = trips × (max(per_sm × (body + trip / resident), rounds × simds × (body + trip) / waves) + rounds × latency / (stages − 1))`，其中 `rounds = ⌈per_sm / resident⌉`，`body` 为重叠后的管线（`Planner::pipelined`）。填充的每个字节按其来源定价：不同字节能放进四分之三缓存时按 L2 速率，超出一半以上时按 DRAM 速率，两者之间取混合，且绝不贵于同时在飞的块彼此重读的那一份——同组的 A 行与各自的 B 列（`Planner::fill_byte`）。没有 wave 那项约束时，gfx1151 的排序以单 warp 的 256×256 tile 为首，而它们只跑出 1–3 TFLOP/s。布局的寄存器计数只用于估计驻留块数；它不决定准入，因为在片段被复制的地方（gfx11 WMMA）它会高估。排序保留一个 tile 的所有流水线；[调优存储](./tuning)在编译后内核不溢出的前 64 个输出 tile 与 warp 网格中挑选，每个取其第一条干净的流水线（全部溢出时：取前 64 个能编译的），然后再在胜者 tile 的其余流水线中挑选。由编译器准入、由计时挑选，是因为模型对块 tile 排序得好、对 warp 网格排序得差（在 sm_86 上实测最快的配置排在 109 个 tile 中的第 92 位）；在胜者确定之前，流水线仅按模型排序，因此会溢出的展开环形缓冲让位于其不展开的孪生配置：在 gfx1151 上，d = 128 时展开的 64×16 和 128×16 注意力 tile 会溢出，而不展开的编译干净。门控权重对模型而言是 `2·bn` 宽的 B tile 加两个累加器，因此它的领先配置宽度减半。设备的 `Planner` 持有其目标、测得的代价以及已排序过的问题（算子在每次调用时规划其形状；对整个格点排序需要一毫秒），进程中每个设备一个规划器。

该模型没有拟合常数：每项代价都是设备自己的，由探针测得。当排序与测量不一致时，以测量为准：把该形状加入 `qwen3_gemm_candidates_probe`，并检查排序遗漏了哪项代价。

## Flash attention {#flash-attention}

在序列优先的 `q [batch, t, heads, d]` 与 `k`、`v [batch, tk, kv_heads, d]` 上计算 `o = softmax(q·kᵀ·scale)·v`。Q 驻留在寄存器中，K/V 通过共享内存环形缓冲流入。在线 softmax 状态 `(m, l, o)` 以 f32 携带，使用 `exp2`。

| 特性 | 实现方式 |
|---|---|
| GQA | 查询头 `h` 读取 KV 头 `h / (heads / kv_heads)` |
| 交叉注意力 | `tk ≠ t` 是视图的属性 |
| 任意 `t`、`tk` | 用边界代替填充：查询 tile 和最后一个键块是带边界的视图，超出 `tk` 的键被掩码 |
| Head 维度 | 48, 64, 128 |
| `AttnMask::causal` | 越过对角线的键块通过迭代次数跳过 |
| `AttnMask::window` | 每个查询周围的 `(left, right)`；窗口外的块被跳过 |
| `AttnMask::key_lens` | `[batch]` i32 有效键数；超出长度的块被跳过 |
| `AttnMask::key_mask` | `[batch, tk]` i32 可见键（行步长向上取整到 8），每个块都读取 |
| `AttnMask::seg_start` | `[batch, t]` i32 打包行的段起点，沿 `t` 非递减 |
| `AttnMask::bias` | 与数据流同类型的 `[batch 或 1, heads, t, tk]` 加性偏置（行步长向上取整到 8），经由 K、V 旁边自己的环形槽流入（准入时其字节与 K、V 的一样计入），在掩码之前加到缩放后的分数上；追加行所在的块直接读取其偏置块 |

掩码按 tile 类别处理。只有被因果、窗口、长度或段边缘穿过的键块才计算谓词（基于块起点的 `select_if`）；完全在内部的块不加掩码运行。bool 键掩码在每个块中应用。看不到任何键的查询得到 NaN，与空行上的 softmax 一致。

**缓存模式**（`AttnSpec::cache = Some(Cache { .. })`）从保存多层头的缓存 `[rows, tk, heads_total, d]` 中读取 K 和 V：

| `Cache` 字段 | 作用 |
|---|---|
| `head_start` | 本注意力的 `kv_heads` 个头在缓存行中的第一个 |
| `row_map` | 一个 `[batch]` i32 参数：每个批次通道读取的缓存行 |
| `appended` | 在缓存前缀之后计分的 `[batch, kv_heads, d]` 键和值（解码器步骤刚投影出的 token） |

**键拆分**（`FaCfg::splits > 1`）为少数长行（例如解码器步骤）提供更多块。每个拆分写出 f32 部分结果 `o_part`、`m_part` 和 `l_part`，由 `combine` 合并。

`FaCfg { bq, bkv, stages, splits, unroll, warp_rows, .. }` 每 `warp_rows` 个查询行使用一个 warp：16（一行原子）或 32（warp 读取的每个键、值片段喂给两行原子的乘积，代价是每 lane 两倍的累加器）。候选来自与 GEMM 相同的格点和代价模型（`Planner::attention_ranked`）：16–128 行的查询块、16–128 个键的键块、目标运行的环形缓冲深度、展开与不展开，只保留能降级的配置（填充能按每线程块拆分，环形缓冲放得进共享内存），并按一个键块的代价排序（每个 warp 读取的 K 和 V 片段、流入的两个 tile、矩阵核步骤、softmax 每个分数的向量工作以及行程的固定开销），覆盖每一轮查询块的所有键块，每轮的约束与 GEMM 相同。寄存器估计是查询、键、分数、概率和值片段之和，它们在编译后的循环中同时存活；配置是否溢出由编译后的内核说了算。超出问题查询数的 warp 会增加其填充行的工作量，因此在 tile 填满设备的地方，解码器步骤以单 warp 块为首；在 tile 让 SIMD 空闲的地方，填充的 warp 在其中一个 SIMD 上运行。因果掩码为每个查询块计入其对角线。在值 tile 的填充经由寄存器的地方，它以列主序存储（参见[布局与降级](./layouts-and-lowering#layouts-and-lowering)），因此其片段像键一样以 16 字节加载收集；在 gfx1201 上测得，这比行主序 tile 每个键一次加载快 5–28%（视形状而定）。在 RDNA4 上，填充本身就是转置加载（`global_load_tr_b128`），因此其寄存器保存列段，也以 16 字节向量存储。

softmax 步骤把向量工作压到最少，这在矩阵核与 SIMD 共用 ALU 的地方（RDNA）最要紧。一行只在其运行最大值以 log2 计增长超过 8 时才采纳新值（FlashAttention-4 的惰性重缩放）：否则修正量恰为 1，而概率保持在 2⁸ 以下。规划器以朴素形式提供每个配置；其余五种（`FaCfg::forms`：`truncate_p`、`skip_rescale`、`raw_max`、`sum_after`、`bias_direct`）是调优器的旋钮，与 tile 轮流调优（`ops::attention::tune`，经由 `descend`），因为某个 tile 可能只在部分形式下才能编译干净：先在排序中第一个编译干净的配置上做一遍形式调优，再在保留下来的形式下对排序做 tile 与拆分选择，然后在该胜者上再做一遍形式调优（由存储支撑，因此没有变化时代价很低）。一遍形式调优是坐标下降：按上述顺序每种形式各一次，做一个两候选的调优，比较目前为止的形式与翻转该形式后的同一配置，只在测得更快时保留翻转（溢出的翻转不被准入）。哪种形式节省寄存器在任何地方都不是规则：由测量决定。关闭调优时，每个注意力都运行朴素形式。启用 `truncate_p` 时，以 bf16 保存的概率截断而非舍入——在舍入需要多条指令的地方，一条指令处理一对——行和按所保存的值相加，因此归一化保持无偏；能一条指令舍入一对的目标（sm_80+、gfx950）则舍入并对 f32 值求和，因为对保存值的额外转换在 sm_86 上代价为 3–5%。启用 `skip_rescale` 时，输出累加器的重缩放是一个 `Scale`，凡所有行都保留了最大值的 wave 都会跳过它；在矢量 ALU 上发射矩阵乘积的地方它都有收益，而在独立矩阵核旁只在满 tile 的每个 lane 持有 64 个输出寄存器时有收益（sm_86：d = 128 时 +4%，d = 64 时持平，解码步骤慢 5–7%，其唯一的一行保留普通乘法）。启用 `raw_max` 时（在查询填满的 RDNA tile 上有收益），运行最大值保持在原始分数的单位中（正的缩放与最大值可交换；有 bias 时保持缩放形式），于是只缩放指数和修正所需的两个行值——每行一次乘法而非每个分数一次——代价是步骤依赖链上多一次乘法，解码步骤的单个 wave 能感觉到（3%），满 tile 则不会。启用 `sum_after` 时，行和跟在乘积之后，而不是在重缩放之前；截断的概率无论哪种顺序都按保存的值求和（在目标具备时用 bf16 对点积），舍入的概率则以 f32 求和。在乘积之后，两个消费者只读取一次打包的截断值（gfx1201）；在乘积之前，在使用舍入的 sm_86 上，该顺序在 d = 128 时快 3%（24.7 对 24.0 TFLOP/s，此项工作之前为 22.4），在 d = 64 和解码步骤上持平。启用 `bias_direct`（仅在有偏置时才会翻转）时，每个键块的偏置块直接从全局内存按分数的布局读入，而不是经由 K、V 旁边自己的环形槽流入（后者是规划的形式）：在 gfx1151 上，其累加器每个 lane 持有一列，从环中按两字节收集该块，无因果掩码时直接读取更快（d = 64、B 4、H 8、T 2048 时 3.0 对 3.6 ms），有因果掩码时环更快（1.88 对 2.12 ms）。在因果掩码下（或看到所有更早块的分块），最后的查询块评分的键最多，它们先运行，于是一轮的尾部是最短的块。在 gfx1201 上，惰性重缩放、截断、跳过与下文的原始 `exp2` 把 d = 128 内核在随机数据上从 90 提到 116 TFLOP/s，Qwen3 的因果注意力从 63 提到 77；原始最大值、对点积与块顺序再把它们提到 120 与 79，d = 128 因果 96 → 100，d = 64 105 → 108（因果 87 → 92），解码步骤不变。相对于循环中仅有 WMMA 时可达的 162 TFLOP/s，剩余部分分布很散：向量工作占 15%，其余是加载、存储、屏障与寻址。

每个 tile 配置都与拆分数交叉组合。`Attn::splits = Some(n)` 固定为 `n`：只有至少有 `n` 个键块的配置参与排序，且以 `n` 排序（上限为任一配置的最大键块数），因此指定的拆分数从不会因键块过大而被丢弃。为 `None` 时，`Planner::splits` 提出 1、2、使每个 SM 保持两个块的附近数值，以及每个键块一个拆分。它只保留不超过键块数与该 SM 目标两倍中较小者的数值。不拆分的配置排在最前，拆分候选与其合并内核一起计时。

注意力吞吐量探针（B 4、H 8、T 2048、bf16）测得的 TFLOP/s，与 tk1 对比：

| 情形 | tk3 | tk1 |
|---|---|---|
| d 64 | 24.1 | 23.0 |
| d 64 因果 | 22.6 | 20.7 |
| d 128 | 22.3 | 22.3 |
| d 128 因果 | 20.9 | 16.5 |

解码探针以 µs 计时一个 Whisper large-v3 解码器步骤的注意力。Whisper 解码器步骤调用的正是这些内核：

| 情形 | tk3 | tk1 |
|---|---|---|
| 自注意力，200 个缓存键 + 追加的一个 | 21.5 | 89.1 |
| 交叉注意力，1500 个共享键 | 66.3 | 78.8（不拆分为 636） |

待解决：解码的 `row_map` 仍被排除在调优形状之外（调优器在每个批次通道一行的临时缓存上测量，此时映射会读到垃圾行），因此调优后的交叉注意力配上其映射未必最快：在 gfx1151 上（2026-10-14），它调优到 90 µs，而不拆分为 60 µs。

## 注意力前处理 {#attention-prologue}

`heads` 把融合投影 `qkv [batch, t, (heads + 2·kv_heads)·d]` 拆分为序列优先的 `q [batch, t, heads, d]` 以及 `k`、`v [batch, t, kv_heads, d]`。可选地，它用 `[d]` 权重在每个头上对 `q` 和 `k` 做 RMS 归一化，然后对它们施加旋转位置编码。旋转把头的两半配对，使用 `[t, d/2]` 的 `(cos, sin)` 表，可共享或按批次提供（`Rope { per_batch }`）。一个块处理一个头槽的 `br` 行。`d` 必须是 16..=256 范围内的 2 的幂，候选为 `br ∈ {4, 8, 16}`。

## 归一化 {#norms}

`rows::norm` 每行使用一个 warp，在 f32 中执行融合的先归约后映射：

| `Norm` | 公式 |
|---|---|
| `Layer` | `(x − mean)·rsqrt(var + eps)·w + b` |
| `Rms` | `x·rsqrt(mean(x²) + eps)·w` |

当 `residual: true` 时，它对舍入到元素类型的 `x + residual` 做归一化，并同时写出该和，即 transformer 层的 pre-norm 残差流。`d` 必须是 256..=2048 范围内的 2 的幂，候选为每块 `br ∈ {4, 8, 16}` 行。在 sm_86 上测得达到内存带宽的 92%。
