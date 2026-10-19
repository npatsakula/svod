---
sidebar_label: 内核库
---

# 内核库

`svod_tk3::kernels` 中的每个内核都是一个函数 `fn k<T: Elem>(spec: &Spec) -> Program`，对 16 位元素类型（`BF16` 或 `F16`）泛型。每个 spec 都有一个 `batch: Batch`：`Batch::Static(n)` 启动 `n` 个批次，`Batch::Var { name, min, max }` 按名称绑定的运行时变量的实际值启动批次，缓冲区按 `max` 分配。每种配置类型都有一个 `lowering(target)`。模型通过[算子层](./op-layer)使用这些内核，由算子层选择配置。

| 内核 | Spec → 程序 | 内核名称 | 算子 |
|---|---|---|---|
| GEMM + 尾处理 | `GemmSpec` → `gemm::gemm` | `gemm` | `ops::linear` |
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

`GemmCfg { tile: [bm, bn, bk], stages, warps: [rows, cols], group_m, unroll }`。超出 `m` 的行和超出 `n` 的列是带边界的视图，因此 `m` 和 `n` 不受限制。`k` 必须是 `bk` 的倍数。

算子层从四个 tile 家族中抽取候选，每个家族都带有在 sm_86 上测得的最佳流水线：

| 家族 | 级数 | Warp 网格 | 展开 | 测量说明 |
|---|---|---|---|---|
| 128×128×32 | 3 | 2×4 | 否 | 4096³ 在此达到峰值：25.4 TFLOP/s |
| 128×64×32 | 2 | 2×2 | 是 | |
| 64×128×32 | 2 | 2×2 | 否 | |
| 64×64×32 | 3 | 2×2 | 是 | M = 704 (Nemotron)：128 行 tile 不敌 64×64 |

`config::gemm_candidates` 以最大的、其网格能让每个 SM 获得 8 个块且填充不超过输出面积 1/16 的家族为首（否则用 64×64）。它加入该家族的变体（另一种级数、`bk = 64`、翻转的 `unroll`、转置的 warp 网格），然后是其他家族。当 `k` 不是 `bk` 的倍数时，`bk` 减半直至 16；环形缓冲超过目标共享内存的配置被丢弃。最多保留八个候选，由[调优存储](./tuning)从中选择。

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

掩码按 tile 类别处理。只有被因果、窗口、长度或段边缘穿过的键块才计算谓词（基于块起点的 `select_if`）；完全在内部的块不加掩码运行。bool 键掩码在每个块中应用。看不到任何键的查询得到 NaN，与空行上的 softmax 一致。

**缓存模式**（`AttnSpec::cache = Some(Cache { .. })`）从保存多层头的缓存 `[rows, tk, heads_total, d]` 中读取 K 和 V：

| `Cache` 字段 | 作用 |
|---|---|
| `head_start` | 本注意力的 `kv_heads` 个头在缓存行中的第一个 |
| `row_map` | 一个 `[batch]` i32 参数：每个批次通道读取的缓存行 |
| `appended` | 在缓存前缀之后计分的 `[batch, kv_heads, d]` 键和值（解码器步骤刚投影出的 token） |

**键拆分**（`FaCfg::splits > 1`）为少数长行（例如解码器步骤）提供更多块。每个拆分写出 f32 部分结果 `o_part`、`m_part` 和 `l_part`，由 `combine` 合并。

`FaCfg { bq, bkv, stages, splits }` 每 16 个查询行使用一个 warp。按 head 维度列出的候选，第一个是未调优时的选择：

| `d` | 候选 `(bq, bkv, stages)` | 测量说明 (sm_86) |
|---|---|---|
| 48、64、128 且 `t ≤ 16` | (16, 64, 2), (16, 64, 3), (16, 32, 2) | 解码器步骤受带宽限制：单 warp 块让每个 SM 能容纳多个块 |
| 48 | (64, 64, 2), (64, 64, 3) | K/V 填充能在块内线程间整除的形状（96 字节行） |
| 64 | (64, 64, 2), (64, 64, 3), (128, 64, 2), (64, 32, 2), (128, 32, 2), (64, 32, 3) | |
| 128 | (64, 32, 2), (64, 32, 3), (128, 32, 2), (64, 64, 2), (128, 64, 2), (128, 32, 3) | `bkv = 64` 测得 18.2 TFLOP/s，`bkv = 32` 为 22.2：每块 64 KB 使每个 SM 只剩一个块 |

每个 tile 配置都与拆分数交叉组合。`Attn::splits = Some(n)` 固定为 `n`，上限为键块数。为 `None` 时，`config::split_candidates` 提出 1、2、使每个 SM 保持两个块的附近数值，以及每个键块一个拆分。它只保留不超过键块数与该 SM 目标两倍中较小者的数值。不拆分的配置排在最前，拆分候选与其合并内核一起计时。

注意力吞吐量探针（B 4、H 8、T 2048、bf16）测得的 TFLOP/s，与 tk1 对比：

| 情形 | tk3 | tk1 |
|---|---|---|
| d 64 | 24.1 | 23.0 |
| d 64 因果 | 22.6 | 20.7 |
| d 128 | 22.3 | 22.3 |
| d 128 因果 | 20.9 | 16.5 |

解码探针以 µs 计时一个 Whisper large-v3 解码器步骤的注意力。内核按该模型的形状运行，但 Whisper 模型尚未调用它们：

| 情形 | tk3 | tk1 |
|---|---|---|
| 自注意力，200 个缓存键 + 追加的一个 | 21.5 | 89.1 |
| 交叉注意力，1500 个共享键 | 66.3 | 78.8（不拆分为 636） |

## 注意力前处理 {#attention-prologue}

`heads` 把融合投影 `qkv [batch, t, (heads + 2·kv_heads)·d]` 拆分为序列优先的 `q [batch, t, heads, d]` 以及 `k`、`v [batch, t, kv_heads, d]`。可选地，它用 `[d]` 权重在每个头上对 `q` 和 `k` 做 RMS 归一化，然后对它们施加旋转位置编码。旋转把头的两半配对，使用 `[t, d/2]` 的 `(cos, sin)` 表，可共享或按批次提供（`Rope { per_batch }`）。一个块处理一个头槽的 `br` 行。`d` 必须是 16..=256 范围内的 2 的幂，候选为 `br ∈ {4, 8, 16}`。

## 归一化 {#norms}

`rows::norm` 每行使用一个 warp，在 f32 中执行融合的先归约后映射：

| `Norm` | 公式 |
|---|---|
| `Layer` | `(x − mean)·rsqrt(var + eps)·w + b` |
| `Rms` | `x·rsqrt(mean(x²) + eps)·w` |

当 `residual: true` 时，它对舍入到元素类型的 `x + residual` 做归一化，并同时写出该和，即 transformer 层的 pre-norm 残差流。`d` 必须是 256..=2048 范围内的 2 的幂，候选为每块 `br ∈ {4, 8, 16}` 行。在 sm_86 上测得达到内存带宽的 92%。
