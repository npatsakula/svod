---
sidebar_label: 测试与调试
---

# 测试与调试

每个 tk3 内核都在两个层面上检查。在主机上，其 tile 程序在解释器中运行，并与直接参考实现比较。在 CUDA 设备上，降级后的内核与解释器或该算子自身的计算图回退比较。测试在容差范围内比较数值，而不是比较 IR 的哈希。

## 解释器 {#the-interpreter}

`interp::run(&program, params, vars)` 在主机上执行 tile 程序。它为每个参数接收一个 `Vec<f64>`，并返回运行后的所有参数。每个块按顺序执行其语句，流水线以其串行交错的方式运行。每个值都会舍入到其元素类型（`interp::round_to`），因此结果就是正确的降级在累加顺序误差之内必须重现的值。运行时变量按名称绑定：

```rust
let out = run(&prog, vec![a.clone(), b.clone(), vec![0.0; m * n]], &[("b", 1)]).unwrap();
```

它返回 `interp::Error::{UnboundVar, ParamSize, Raw}`。它不模拟线程和屏障，因此顺序错误只会在设备上暴露。

## 测试文件 {#test-files}

| 文件 | 层面 | 检查内容 |
|---|---|---|
| `layout.rs`, `layouts.rs`, `atoms.rs` | 主机 | F2 代数定律、每个原子与其闭式公式的对比、推断结果 |
| `interp.rs`, `schedule.rs`, `build.rs` | 主机 | 解释器语义、流水线展开 |
| `ops_plan.rs` | 主机 | 每个算子在各种形状、数据类型和目标下的 `Plan` 与错误 |
| `attention.rs`, `heads.rs`, `rows.rs` | 主机 + 设备 | 程序与 f64 参考比较，然后降级后的内核与程序比较 |
| `device.rs`, `parts.rs` | 设备 | GEMM（算子层可能选择的每个配置）以及注意力所组合的各部件，与解释器比较 |
| `ops.rs` | 设备 | 每个算子在刁钻形状上与其计算图回退比较，调优关闭 |
| `tune.rs` | 主机 + 设备 | 存储往返、键稳定性、一次真实的 GEMM 调优 |

除非 `SVOD_DEVICE` 指定了 CUDA 设备，设备测试会输出一行 `skipped: no CUDA device` 并跳过。在 GPU 上串行运行整个测试套件：

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk3 --lib --release -- --test-threads=1
```

Nemotron 的模型级关卡（真实权重、生成的金标准）：

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-model --release --lib nemotron_diar::parity::half_precision -- --ignored --nocapture --test-threads=1
```

## 内核运行了吗？ {#did-the-kernel-run}

发生回退的算子构建的是计算图算子而不是 `CALL`。`ops.rs` 的测试在结果的调用中查找内核名称：

```rust
fn assert_kernel(t: &Tensor, name: &str) {
    let calls: Vec<String> = t
        .uop()
        .toposort()
        .iter()
        .filter_map(|u| match u.op() {
            Op::Call(ops::Call { info, .. }) => info.name.clone(),
            _ => None,
        })
        .collect();
    assert!(calls.iter().any(|n| n == name), "no {name} kernel among {calls:?}");
}
```

要在没有设备的情况下预测走哪条路径，可以调用 `ops::shape` 规划器（参见[算子层](./op-layer#kernel-or-graph)）。

## 环境变量 {#environment-variables}

| 变量 | 作用 |
|---|---|
| `TK3_DUMP_LIST=1` | 程序降级时打印每条发射的指令（`[i] id op dtype <- sources`）。主体会被记忆化缓存，因此只打印每个程序的第一次降级 |
| `SVOD_SPEC_DEBUG=1` | 程序未通过 IR spec 校验时，打印被拒绝的指令及其树 |
| `SVOD_TK3_TUNE=0` | 不做测量；运行第一个候选（参见[调优](./tuning)） |
| `SVOD_DEVICE=CUDA:0` | 在 GPU 上运行（默认设备决定内核是否适用） |

## 剖析模型 {#profiling-a-model}

Nemotron 示例先预热一次（同时填充调优存储），计时一次运行，然后打印第三次运行的逐内核报告：

```bash
cargo run -p svod-model --release --example nemotron_diarize -- audio_1.wav --dtype bf16 --profile
```

tk3 内核以其内核名称（`gemm`、`flash_attention`、`heads`、`layer_norm`……）与图内核并列出现，并以相同方式计时。

:::tip[在安静的 GPU 上测量]
RTX 3060 空闲时运行在 210 MHz，GPU 上的其他进程会使每个数字产生偏差。探针和调优存储会先拉升时钟。运行探针时使用 `--test-threads=1`。
:::
