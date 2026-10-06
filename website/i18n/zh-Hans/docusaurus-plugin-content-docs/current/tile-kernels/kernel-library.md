---
sidebar_label: 内核库
---

# 内核库

USE 面孔：`svod-tk` 提供的每个内核都能直接用普通张量调用，无需了解任何分块知识。每个内核都返回一个惰性 `Tensor`（一个 `Op::Call` 节点），它能组合进模型图，并通过常规的 `prepare()` 路径实现；每个内核也都遵循 [向 IR 中编写](./lowering) 中的三向契约：

| 结果 | 含义 |
|---|---|
| `Ok(Some(out))` | 内核已运行 |
| `Ok(None)` | 内核不适用：设备不在该内核的 `ArchSet` 中、缺少其 LLVM 后端，或形状无法分块——调用方应有意识地回退 |
| `Err(LaunchError)` | 请求本身有误（dtype、秩、符号维度、整除规则）——属于调用方的 bug |

除非另有说明，操作数为 bf16 或 f16，累加为 f32。

---

## 目标架构 {#targets}

每个内核都声明自己的 `ArchSet`（`tk/src/target.rs`）：一份显式的 AMD 列表，加上一个开放式的 CUDA 计算能力下限和一个 Apple GPU 家族下限。

| 内核 | gfx942 (CDNA3) | gfx1151 (RDNA3.5) | gfx1200 / gfx1201 (RDNA4) | CUDA sm_80+ | Metal Apple7+ |
|---|---|---|---|---|---|
| `flash_attention` / `_with` / `_tuned` | 是 | 是 | 是 | 是 | 是 |
| `matmul`（方阵） | 是 | 是 | 是 | 是 | 是 |
| `gemm_nt` / `_with` / `_with_epilogue` | — | 是 | 是 | 是 | — |
| `rms_norm` / `add_rms_norm` | — | 是 | 是 | 是 | — |
| `single_query_attention` / `_packed` | 是 | 是 | 是 | 是 | — |
| `knn` | 是 | 是 | 是 | — | — |
| `kmeans_assign` | 是 | 是 | 是 | — | — |

对应的常量是 `tk/src/kernels/` 中的 `FA_SUPPORTED_ARCHS`、`MATMUL_SUPPORTED_ARCHS`、`GEMM_NT_SUPPORTED_ARCHS`、`NORM_SUPPORTED_ARCHS`、`SQ_ATTENTION_SUPPORTED_ARCHS`、`KNN_SUPPORTED_ARCHS` 和 `KMEANS_SUPPORTED_ARCHS`。一个架构家族要加入某个内核，靠的是验证，以及测出它自己的分块表——`gemm_nt` 和各个 norm 只支持 wave32，是因为还没人为它们测过 wave64 的分块表，而不是因为函数体在那里跑不了。`flash_attention_supported(&device)` 只回答架构门控这一问，供那些需要在启动前填充或分桶序列长度的调用方使用。

---

## Flash attention

```rust
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> LaunchResult<Option<Tensor>>
pub fn flash_attention_with(q, k, v, opts: FaOpts) -> LaunchResult<Option<Tensor>>
pub fn flash_attention_tuned(q, k, v, opts, policy: impl Fn(&DeviceSpec, GpuArch) -> FaPolicy + Copy) -> ..

pub struct FaOpts<'a> {
    pub causal: bool,                      // default true
    pub key_lens: Option<&'a Tensor>,      // [B] i32 valid-key counts: keys >= key_lens[b] are masked
    pub seg_start: Option<&'a Tensor>,     // [B, N] i32: query q of batch b sees no key before seg_start[b, q]
}
```

`q` 为 `[B, N, H, D]`，`k`/`v` 为 `[B, N, H_kv, D]`（GQA：`H % H_kv == 0`），输出为操作数 dtype 的 `[B, N, H, D]`。布局是序列优先而非头优先——模型可以把投影结果直接 reshape 进来，无需转置。

- `Ok(None)`：不在架构集合内；`N` 不是 `q_blk · 8` 的倍数（每个 warp 的 Q 分块乘以一个工作组的八个 wave；`FLASH_ATTENTION_SEQUENCE_MULTIPLE` 是基线的 `128`）；KV 长度与 `N` 不同（交叉注意力尚未实现）；某个头维度使双缓冲的 K/V 分块超出设备的共享内存。
- `Err`：dtype 不在 `{bf16, f16}` 之内，或 `q` 与 `k`/`v` 的 dtype 不一致；`D % 16 != 0`；`H % H_kv != 0`；`k`/`v` 的形状不是 `[B, N, H_kv, D]`。

`key_lens` 只屏蔽键——填充出来的查询行照样会被计算，由调用方丢弃。`key_lens[b] == 0` 会被钳制为 `1`，以保证该行保持有限值。`seg_start` 把多条序列打包进同一行：每个条目必须落在 `0..=q` 内，并至少留下一个可见的键。[Flash Attention](./flash-attention) 是完整的演练示例；每个 warp 的分块在首次使用时测量得出（[自动调优](./tuning)）。

---

## GEMM

```rust
pub fn matmul(a: &Tensor, b: &Tensor) -> LaunchResult<Option<Tensor>>              // [n, n] · [n, n] → f32
pub fn gemm_nt(x: &Tensor, w: &Tensor) -> LaunchResult<Option<Tensor>>             // [lead..., K] · [N, K]ᵀ → [lead..., N]
pub fn gemm_nt_with(x, w, cfg: impl Fn(usize, usize, usize) -> Option<GemmCfg> + Copy) -> ..
pub fn gemm_nt_with_epilogue(x, w, epilogue: Epilogue<&Tensor>) -> LaunchResult<Option<Tensor>>

pub enum Epilogue<T> {
    Plain,               // y = x·wᵀ
    Add(T),              // y = x·wᵀ + residual, residual [lead..., N] in the operand dtype
    SwiGlu { pair: usize }, // y = silu(gate)·up off a fused [2I, K] gate/up weight; y is [lead..., N/2]
}
pub fn swiglu_pair_width(spec: &DeviceSpec) -> Option<usize>
```

`matmul` 是方阵参考内核：输入可以是任意浮点 dtype（转换为 bf16），输出 f32，支持所有架构。它是 DSL 的性能哨兵，而不是生产用的 GEMM。

`gemm_nt` 才是生产用的线性层。`x` 为任意秩 ≥ 2 的 `[lead..., K]`（`[B, L, K]` 激活无需 reshape 或拷贝即可绑定），`w` 按权重的存储方式为 `[N, K]`，`y` 为操作数 dtype 的 `[lead..., N]`——f32 累加器在寄存器中完成窄化，因此不会经由内存往返一次 f32。`M = ∏lead` 和 `N` 必须是 64 的倍数，`K` 必须是 32 宽条带的倍数且至少有两条；否则返回 `Ok(None)`，由调用方填充到 128 或改用 `Tensor::linear`。

尾声（epilogue）正是这个内核存在的理由：它们把图在 GEMM 之后本要付出的那一遍计算折叠进 GEMM 的存储中。`Add` 在存储自身的偏移处读取残差，并以输出 dtype 相加，舍入方式与图中的 `try_add` 完全相同。`SwiGlu` 要求融合权重的行按 `pair` 行一组、gate/up 交替排列——`swiglu_pair_width(&device)` 等于设备分块表中的 `reg_n / 2`，若各分块不一致则为 `None`（调用方保留一个独立的 SwiGLU 过程）。模型在加载时一次性按该顺序排好权重，因为 `M` 在启动时才选定分块，每个候选分块都必须读取同一种排列。

分块来自 `GemmPolicy`（`tk/src/kernels/gemm.rs`）：一张按架构家族划分的表——`CUDA_TILES`、`RDNA_TILES`、`RDNA4_TILES`——对每个形状和尾声在首次使用时测量（[自动调优](./tuning)），关闭调优时则采用静态的 `GemmPolicy::cfg` 选择。`gemm_nt_with` 由调用方提供选择器（基准测试就是这样扫描的）。

---

## RMS norm

```rust
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> LaunchResult<Option<Tensor>>
pub fn add_rms_norm(x, residual: &Tensor, weight, eps) -> LaunchResult<Option<(Tensor, Tensor)>>   // (h, y)
pub fn select_norm_cfg(rows: usize, d: usize, lanes: usize) -> Option<NormCfg>
```

`x` 为 `[rows..., D]`，`weight` 为 `[D]`，二者同为一种 16 位 dtype。每行一个 wave，整行驻留在寄存器中，平方和由蝶形 shuffle 完成——没有 LDS，没有屏障，也没有 `RANGE`。数值逐操作地与图保持一致：`y = dtype((f32(x) · rsqrt(Σx²/D + eps)) · f32(w))`，只在最后舍入一次；唯一不同的是求和顺序。当 `D` 不是 wave 宽度的倍数，或每个 lane 超过 64 个元素（wave32 下为 `2048`）时，返回 `Ok(None)`。

`add_rms_norm` 返回 `(h, y)`，其中 `h = x + residual` 的舍入方式与图中的加法相同，`y = rms_norm(h)`，这样一个 pre-norm 解码器层只需写、读一次残差流。若前面的投影已通过 `Epilogue::Add` 接收了残差，则两遍的 `rms_norm` 就够了；`model/src/qwen3/decoder_layer.rs` 按层在两者之间选择。

---

## 单查询注意力

```rust
pub fn single_query_attention(q, k, v, opts: SqAttentionOpts<'_>) -> LaunchResult<Option<Tensor>>
pub fn single_query_attention_packed(q, k, v, head_offset: usize, opts) -> LaunchResult<Option<Tensor>>

pub struct SqAttentionOpts<'a> {
    pub key_lens: Option<&'a Tensor>,                 // [B] i32, entries in 0..=N
    pub include_last: bool,                           // also score key N-1 (Whisper's self-cache slot)
    pub appended: Option<(&'a Tensor, &'a Tensor)>,   // the step's own [B, 1, H, D] K/V, scored after the prefix
    pub split: Option<usize>,                         // K/V chunks; None = the device's SqPolicy, tuned on first use
    pub cache_map: Option<&'a Tensor>,                // [B] i32: which K/V row each query row reads
}
```

这是解码步内核：`q` 为 f32 的 `[B, 1, H, D]`，`k`/`v` 为 f32、f16 或 bf16 的 `[B, N, H_total, D]`（或 `[1, N, H_total, D]`，用一份缓存服务所有行），输出为 f32 的 `[B, 1, H, D]`。
一个 wave 负责一个 `(batch, head)`；`Q` 留在寄存器中，K/V 沿 `N` 流过；点积是 XOR-shuffle 全归约，softmax 是一遍式在线更新。没有 LDS，也不用矩阵核心，所以它的 `ArchSet` 是最宽的 AMD 列表外加 CUDA。

`_packed` 从打包缓存中选取 `head_offset..head_offset + H` 这些头，而无需对缓存切片。较长的无掩码注意力会把 K/V 切成连续的块，每块一个 wave，再在第二遍中合并它们的 softmax 状态；`SqPolicy` 根据设备的驻留 wave 预算确定切分数，并在首次使用时测量最接近的几个约数。

---

## k-NN 与 k-means

```rust
pub fn knn(x: &Tensor, c: &Tensor, k: usize) -> LaunchResult<Option<(Tensor, Tensor)>>           // (dists [N, k] f32, idxs [N, k] i32)
pub fn kmeans_assign(x: &Tensor, c: &Tensor) -> LaunchResult<Option<(Tensor, Tensor)>>         // (cluster_ids [N] i32, best_dist [N] f32)
pub fn kmeans_update(x, cluster_ids, old_centroids) -> LaunchResult<(Tensor, Tensor)>           // (new_centroids [K, D], shift [K])
```

二者都让语料（质心）流经矩阵核心，并基于不含 x² 的分数 `‖c‖² − 2⟨x, c⟩` 维护一个运行中的 top-K（argmin），因此从不构造 `[N, M]` 距离矩阵；主机端负责转换为 bf16、把 `D`（以及 `N`）填充到 WMMA 边长，并加回 `‖x‖²` 以得到精确的 f32 距离。`knn` 的 `k` 取值范围为 `1..=16`。`kmeans_update` 是纯图操作——按簇的 `scatter_reduce`、空簇修补、按簇位移——因为排序/散射模式无法分块；Lloyd 循环由调用方负责。仅支持 AMD。

---

## 在模型中使用内核

策略——用哪个内核、怎么回退——属于模型，永远不属于内核。`model/src/qwen3/` 中的 Qwen3 解码器是参考集成：

```rust
// model/src/qwen3/linear.rs
pub(crate) fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    if fusable(x, w)
        && let Some(y) = svod_tk::gemm_nt(x, w).context(TkSnafu)?
    {
        return Ok(y);
    }
    Ok(x.contiguous().linear().weight(w).call()?)
}
```

```rust
// model/src/qwen3/attention.rs
if matches!(q.dtype().base(), ScalarDType::Float16 | ScalarDType::BFloat16)
    && let Some(out) =
        svod_tk::flash_attention_with(q, k, v, svod_tk::FaOpts { causal: true, key_lens: None, seg_start })
            .context(TkSnafu)?
{
    return Ok(out);
}
// else: permute to head-major and run scaled_dot_product_attention
```

值得照搬的三个习惯：

- **先用 `fusable` 做门控。** `Err` 表示请求有误，在调用方一侧做一次 16 位检查，就能避免把合法的 f32 路径报成 bug。
- **桥接错误。** `LaunchError` 被装箱进模型的错误枚举（`#[snafu(source(from(svod_tk::LaunchError, Box::new)))]`），因此构建失败会成为一个附带内核上下文的模型错误。
- **在上游按内核的要求整形。** `embed.rs` 把填充后的序列长度分桶到 `FLASH_ATTENTION_SEQUENCE_MULTIPLE`，`feed_forward.rs` 在加载时按 `swiglu_pair_width` 交错 gate/up 权重，于是内核得以生效而不是拒绝。

了解某个模型内存布局的融合应当放在模型旁边：`model/src/qwen3/tk/mod.rs` 是一个 QKV-norm-RoPE 前奏，用 norm 内核的行词汇编写，由 `NORM_SUPPORTED_ARCHS` 门控，并通过 `graph_launch_multi` 以三个输出启动。同样的 `launch_custom` 策略也适用于它——除了所在位置之外，它在各方面都是一个 tk 内核。
