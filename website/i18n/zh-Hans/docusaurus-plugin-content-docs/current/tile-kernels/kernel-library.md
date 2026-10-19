---
sidebar_label: 内核库
---

# 内核库 {#kernel-library}

`svod_tk3::kernels` 中的每个内核都是一个函数 `fn k<T: Elem>(spec: &Spec) -> Program`，对 16 位元素类型（`BF16` 或 `F16`）泛型。每个 spec 都有一个 `batch: Batch`：`Batch::Static(n)` 启动 `n` 个批次，`Batch::Var { name, min, max }` 按名称绑定的运行时变量的实际值启动批次，缓冲区按 `max` 分配。每种配置类型都有一个 `lowering(&device)`（一个 `atoms::Device`：目标及其代价）。模型通过[算子层](./op-layer)使用这些内核，由算子层选择配置。

| 内核 | Spec → 程序 | 内核名称 | 算子 |
|---|---|---|---|
| GEMM + 尾处理 | `GemmSpec` → `gemm::gemm` | `gemm` | `ops::linear` |
| 隐式 GEMM 卷积 | `ConvSpec` → `conv::conv`（拆分时另加 `gemm::split_merge`） | `conv`、`split_merge` | `ops::conv2d` |
| Flash attention 前向 | `AttnSpec` → `attention::attention` | `flash_attention` | `ops::attention` |
| 拆分合并 | `CombineSpec` → `attention::combine` | `combine_splits` | 带拆分的 `ops::attention` |
| 注意力前处理 | `HeadsSpec` → `heads::heads` | `heads` | `ops::heads` |
| LayerNorm / RMSNorm | `NormSpec` → `rows::norm` | `layer_norm` / `rms_norm` | `ops::{layer_norm, rms_norm, add_*}` |

## GEMM {#gemm}

`c = scale·act(a·bᵀ + bias) + residual`，其中 `a [batch·m, k]`、`b [n, k]`。矩阵乘之后的所有计算都在 f32 累加器上进行，结果在存储时只舍入一次。

| `Epilogue` 字段 | 作用 |
|---|---|
| `bias` | 加到累加器上的 `[n]` 行（门控时为 `[2n]`） |
| `act` | `Act::None`、`Act::Gelu`（erf 近似，误差至多 1.5e-7）、`Act::Silu` |
| `gated` | `b` 为 `[2n, k]`，门控行在上、up 行在下；输出 `act(gate)·up`（SwiGLU、GeGLU） |
| `scale` | 在加残差之前乘以激活后的值 |
| `residual` | 最后加上的 `[batch·m, n]` |
| `out_f32` | f32 输出与残差，累加器不经舍入直接存储 |

`GemmCfg { tile: [bm, bn, bk], stages, warps: [rows, cols], group_m, unroll, core, crews }`。超出 `m` 的行和超出 `n` 的列是带边界的视图，因此 `m` 和 `n` 不受限制。`k` 必须是 `bk` 的倍数。`core: Core::Group` 在 `wgmma` 上发出乘积（sm_90）；`crews: 2` 让两行 warp 在同一个槽上以乒乓方式运行（CDNA）。

不存在 tile 表。`config::Planner::gemm_ranked` 枚举 tile、`bk`、warp 网格、环形缓冲深度和展开方式，保留降级能在目标上运行的配置（原子能铺满 warp 网格，填充能按每线程整 16 字节块拆分，环形缓冲放得进共享内存），再依据设备测得的 `Costs`，以 SM 时间的纳秒数为其余配置排序。排序所用的每一项设备输入都是测得的；其少数几个模型常数（`ops::config` 中的 `BAND`、`CACHE_SHARE`、`CACHE_SPILL`、`OVERLAP_WAVES`）旨在对所有设备成立。排序只负责排列顺序，调优器测量编译后不溢出的前 64 个 tile（[调优](./tuning)）。

## 卷积 {#convolution}

`conv::conv` 是在通道在后的 `x [b, h, w, cin]` 和 `w [cout, kh, kw, cin]` 上的隐式 GEMM：A 是图像的按行映射视图（`conv::im2col_view`），因此填充行读为零，且不物化任何东西。`K = kh·kw·cin` 作为一条流水线运行，每一步都位于同一个抽头之内（`bk` 整除 `cin`）。在只有单个静态批次、网格不饱和的情况下，`ConvCfg::split` 把 K 拆分到网格的 z 维上，由 `gemm::split_merge` 施加尾处理。`ops::conv2d` 把 1×1 卷积当作 `linear` 处理，而把 `groups ≠ 1`、`cin` 不是 16 的倍数以及 `cout` 不是 8 的倍数的情况留给计算图。

## Flash attention {#flash-attention}

在序列优先的 `q [batch, t, heads, d]` 与 `k`、`v [batch, tk, kv_heads, d]` 上计算 `o = softmax(q·kᵀ·scale + bias)·v`。Q 驻留在寄存器中，K/V 通过共享内存环形缓冲流入。在线 softmax 状态 `(m, l, o)` 以 f32 携带，使用 `exp2`。

| 特性 | 实现方式 |
|---|---|
| GQA | 查询头 `h` 读取 KV 头 `h / (heads / kv_heads)` |
| 交叉注意力 | `tk ≠ t` 是视图的属性 |
| 任意 `t`、`tk` | 用边界代替填充：查询 tile 和最后一个键块是带边界的视图，超出 `tk` 的键被掩码 |
| Head 维度 | 任何能被两个乘积铺满、且填充能按每线程整块拆分的 `d`（`FaCfg::fits`）；模型使用 48、64 和 128 |
| `AttnMask::causal` | 越过对角线的键块通过迭代次数跳过 |
| `AttnMask::window` | 每个查询周围的 `(left, right)`；窗口外的块被跳过 |
| `AttnMask::chunk` | 分块注意力：查询能看到自己所在的块、之前 `left` 个块（不设界限时为全部）以及之后 `right` 个块 |
| `AttnMask::key_lens` | `[batch]` i32 有效键数；超出长度的块被跳过 |
| `AttnMask::key_mask` | `[batch, tk]` i32 可见键（行步长向上取整到 8），每个块都读取 |
| `AttnMask::seg_start` | `[batch, t]` i32 打包行的段起点，沿 `t` 非递减 |
| `AttnMask::bias` | 与数据流同类型的 `[batch or 1, heads, t, tk]` 加性偏置（行步长向上取整到 8），在掩码之前加到缩放后的分数上 |

只有被因果、窗口、分块、长度或段边缘穿过的键块才计算谓词；完全在内部的块不加掩码运行。键掩码和偏置在每个块中都会读取。看不到任何键的查询得到零，与计算图的写法一致。

**缓存模式**（`AttnSpec::cache = Some(Cache { .. })`）从保存多层头的缓存 `[rows, tk, heads_total, d]` 中读取 K 和 V：

| `Cache` 字段 | 作用 |
|---|---|
| `rows`、`heads_total` | 缓存的形状 |
| `row_map` | 一个 `[batch]` i32 参数：每个批次通道读取的缓存行 |
| `appended` | 在缓存前缀之后计分的 `[batch, appended, kv_heads, d]` 键和值行（解码器步骤刚投影出的 token），至多一个键块 |
| `app_lens` | 一个 `[batch]` i32 参数：可见的追加行数 |

本注意力的第一个头由一个 `[1]` i32 参数 `head_start` 给出，因此每一层都运行同一个程序。

**键拆分**（`FaCfg::splits > 1`）为少数长行（例如解码器步骤）提供更多块。每个拆分写出 f32 部分结果 `o_part`、`m_part` 和 `l_part`，由 `combine` 合并。`Attn::splits = Some(n)` 固定拆分数；为 `None` 时，`Planner::splits` 提出 1、2、使每个 SM 保持约两个块的附近数值，以及每个键块一个拆分，调优器把每个拆分候选与其合并内核一起计时。

`FaCfg { bq, bkv, stages, splits, unroll, forms, key_warps, warp_rows, core }`：

| 字段 | 作用 |
|---|---|
| `warp_rows` | 每个 warp 的查询行数：16（一行原子）或 32（每个 K、V 片段喂给两行原子）；32 行形式是一个变体，调优器把它与其 16 行 tile 一并计时 |
| `key_warps` | 单行 warp 的块（`bq = 16`，在 `t ≤ 16` 时提供）把其键块拆分到 1、2 或 4 个 warp 上，每个 warp 有自己的环形槽和 softmax 状态，通过共享内存合并 |
| `core` | `Core::Group` 在 `wgmma` 上运行两个乘积（sm_90） |
| `forms` | 下文的 softmax 形式 |

候选来自与 GEMM 相同的格点和代价模型（`Planner::attention_ranked`）。在填充经由寄存器的地方，值 tile 以列主序存储，使其片段以 16 字节加载收集；在 RDNA4 上，填充本身就是转置加载（`global_load_tr_b128`）。

softmax 只在运行最大值以 log2 计增长超过 8 时才采纳新的最大值（FlashAttention-4 的惰性重缩放）。五种 `Forms` 是调优器的旋钮，按规划全部关闭；调优器在测得更快的地方翻转每一种（先做一遍形式调优，再在这些形式下调优 tile，然后在胜者上再调优一次形式）：

| `Forms` 字段 | 作用 |
|---|---|
| `truncate_p` | bf16 概率截断而非舍入，行和按所保存的值相加 |
| `sum_after` | 行和跟在 P·V 乘积之后，而不是在重缩放之前 |
| `skip_rescale` | 输出的重缩放是一个 `Scale`，凡所有行都保留了最大值的 wave 都会跳过它 |
| `raw_max` | 运行最大值保持在原始分数的单位中：每行一次乘法而非每个分数一次 |
| `bias_direct` | 每个键块的偏置 tile 按分数的布局直接从全局内存读取，而不是经由环形槽 |

## 注意力前处理 {#attention-prologue}

`heads` 把融合投影 `qkv [batch, t, (heads + 2·kv_heads)·d]` 拆分为序列优先的 `q [batch, t, heads, d]` 以及 `k`、`v [batch, t, kv_heads, d]`。可选地，它用 `[d]` 权重在每个头上对 `q` 和 `k` 做 RMS 归一化，然后对它们施加旋转位置编码。旋转把头的两半配对，使用 `[t, d/2]` 的 `(cos, sin)` 表，可共享或按批次提供（`Rope { per_batch }`）。一个块处理一个头槽的 `br` 行。`d` 必须是 16..=256 范围内的 2 的幂，候选为 `br ∈ {4, 8, 16}`。

## 归一化 {#norms}

`rows::norm` 每行使用一个 warp，在 f32 中执行融合的先归约后映射：

| `Norm` | 公式 |
|---|---|
| `Layer` | `(x − mean)·rsqrt(var + eps)·w + b` |
| `Rms` | `x·rsqrt(mean(x²) + eps)·w` |

当 `residual: true` 时，它对舍入到元素类型的 `x + residual` 做归一化，并同时写出该和，即 transformer 层的 pre-norm 残差流。`d` 必须是 256..=2048 范围内的 2 的幂，候选为每块 `br ∈ {4, 8, 16}` 行。
