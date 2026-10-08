---
sidebar_label: 算子层
---

# 算子层

`svod_tk3::ops` 是模型调用的接口。每个算子都返回惰性 `Tensor`。当设备、数据类型和形状适合某个内核时，由 tk3 内核计算；否则算子自行构建等价的计算图。模型从不检查内核是否适用，也从不填充到 tile 大小。

```rust
pub fn linear(x: &Tensor, w: &Tensor, opts: Linear) -> Result<Tensor>;
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, opts: Attn) -> Result<Tensor>;
pub fn heads(qkv: &Tensor, opts: Qkv) -> Result<(Tensor, Tensor, Tensor)>;
pub fn layer_norm(x: &Tensor, w: &Tensor, b: Option<&Tensor>, eps: f64) -> Result<Tensor>;
pub fn add_layer_norm(x: &Tensor, residual: &Tensor, w: &Tensor, b: Option<&Tensor>, eps: f64)
    -> Result<(Tensor, Tensor)>;
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor>;
pub fn add_rms_norm(x: &Tensor, residual: &Tensor, w: &Tensor, eps: f64) -> Result<(Tensor, Tensor)>;
pub fn supported(device: &DeviceSpec) -> bool;
```

| 算子 | 形状 | 选项 |
|---|---|---|
| `linear` | `x [lead..., K]`, `w [N, K]` → `[lead..., N]` | `Linear { bias, act, gated, residual }`；门控的 `w` 为 `[2N, K]` |
| `attention` | `q [B, T, H, D]`, `k`/`v [B, Tk, H_kv, D]` → `[B, T, H, D]` | `Attn { causal, keys, window, seg_start, cache, splits, scale }` |
| `heads` | `qkv [B, T, (H + 2·H_kv)·D]` → `q`, `k`, `v` | `Qkv { heads, kv_heads, head_dim, q_norm, k_norm, eps, rope }` |
| `layer_norm`, `rms_norm` | `x [..., D]`, `w`/`b [D]` | `add_*` 接收与 `x` 同形的 `residual`，返回 `(x + residual, norm)` |

`Attn::keys` 可以是 `KeyMask::None`、`KeyMask::Lens(&lens)`（`[B]` 有效键数）或 `KeyMask::Bool(&mask)`（`[B, Tk]`，被关注处为 true）。`Attn::cache` 接收 `Cache { head_start, kv_heads, row_map, appended }`，而 `appended` 要求 `KeyMask::Lens`。`Attn::scale` 默认为 `1/√D`。`Qkv::rope` 是 `[1, T, 1, D/2]`（按位置）或 `[B, T, 1, D/2]`（按 token）的 `(cos, sin)`。掩码的说明见[注意力内核](./kernel-library#flash-attention)。

## 内核还是计算图 {#kernel-or-graph}

决策是 `ops::shape` 中的纯函数 `fn(target, dtypes, extents, …) -> Plan<Cfg>`，因此可以在没有 GPU 的主机上测试。`Plan::Kernel(candidates)` 列出配置，未调优时的选择排在第一。`Plan::Graph(Fallback)` 说明为什么运行计算图。

| `Fallback` | 何时出现 |
|---|---|
| `Target` | 张量不在默认设备上，或设备没有 tk3 配置表（目前：除 CUDA sm_80+ 之外的一切） |
| `Dtype` | 任一操作数为 f32，或操作数没有共享同一种带矩阵核心的 16 位类型 |
| `Symbolic` | 除绑定的首维之外还有符号维度（对 `linear` 而言，符号化的 `N` 也算） |
| `Shape` | `linear`：`N` 不是 8 的倍数或没有行。`attention`：`D ∉ {48, 64, 128}` 或存在空维度。`heads`：`D` 不是 16..=256 范围内的 2 的幂。归一化：`D` 不是 256..=2048 范围内的 2 的幂 |
| `Config` | 没有合适的 tile 配置。对 `linear` 而言，`K` 不是 16 的倍数 |

`test/unit/ops_plan.rs` 中的真实用例，基于 sm_86 目标：

```rust
#[test_case(&[4096, 4096], 4096, false, Ok(BIG); "large grid, deepest ring")]
#[test_case(&[8, 37, 512], 512, false, Ok(SMALL); "medium grid")]
#[test_case(&[37, 40], 96, false, Err(Fallback::Config); "k off every bk")]
#[test_case(&[37, 64], 100, false, Err(Fallback::Shape); "n not a multiple of 8")]
fn linear_plans(x: &[usize], n: usize, gated: bool, want: Result<GemmCfg, Fallback>) {
    assert_eq!(first(shape::linear(Some(&sm86()), &[BF16, BF16], Some(&ext(x)), n, gated)), want);
}
```

:::note[为什么 f32 保留计算图]
内核只接受 16 位操作数。把 f32 模型向下转换会用大约三位十进制精度换取速度，因此算子层把这个选择留给模型的数据类型。
:::

## 批次变量 {#batch-variables}

当操作数的第 0 维绑定到运行时变量时，它可以是符号维度。该变量可以是 JIT 的 `batch_var`，也可以是带最小值和最大值的 `DefineVar`/有界 `Param`。此时内核在网格 z 维上按实际数量启动，缓冲区按最大值分配。输出按容量分配，并在形状中保留符号维度（`Tensor::empty_dynamic`）。因此下一个内核直接绑定已 realize 的缓冲区，不会为了缩小它而产生拷贝。其他任何符号维度都是 `Fallback::Symbolic`。

## 错误 {#errors}

`Err` 只用于计算图算子同样会拒绝的情况，外加内核构建失败：

| `ops::Error` | 含义 |
|---|---|
| `Shape { op, operand, got, expected }` | 某个操作数的形状不适合该算子 |
| `Dtype { op, operand, got, want }` | `w`（或 `k`、`v`、归一化权重、rope 表）的数据类型与输入不同 |
| `Heads { op, heads, kv_heads }` | `heads` 不是 `kv_heads` 的倍数 |
| `Graph { op, source }` | 构建计算图回退失败 |
| `Launch { op, source }` | 降级或绑定内核失败 |

## 在模型中 {#in-a-model}

Nemotron-3-Diarization 的自注意力，摘自 `model/src/nemotron_diar/model.rs`：一次堆叠的 QKV 投影、带 RoPE 的融合前处理、在键长度约束下的注意力，以及在尾处理中加上残差的输出投影。

```rust
fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor), key_lens: &Tensor, residual: &Tensor) -> Result<Tensor> {
    let (b, s, d) = (x.dim(0)?, x.dim(1)?, x.dim_const(2)?);
    let qkv = ops::linear(x, &self.qkv_weight, ops::Linear::default())?;
    let (cos, sin) = rope;
    let split = Qkv {
        heads: self.num_heads,
        kv_heads: self.num_heads,
        head_dim: d / self.num_heads,
        q_norm: None,
        k_norm: None,
        eps: 0.0,
        rope: Some((cos, sin)),
    };
    let (q, k, v) = ops::heads(&qkv, split)?;
    let opts = Attn { keys: KeyMask::Lens(key_lens), ..Attn::default() };
    let out = ops::attention(&q, &k, &v, opts)?;
    project(&self.o_proj, &out.try_reshape([b, s, SInt::Const(d)])?, Act::None, Some(residual))
}
```

`project` 是带有该层偏置、激活函数和可选残差的 `ops::linear`。同一文件中的 LayerNorm 通过 `ops::layer_norm` 运行。
