---
sidebar_label: 调优
---

# 调优

每个算子在 `ops::config` 中为每种形状提供一个候选列表（表格见[内核库](./kernel-library)）。第一个候选是未调优时运行的配置。调优存储在设备第一次遇到某个形状时测量整个列表，并保留胜者。

更好的固定默认值并不够用。Nemotron 的 GEMM（M = 704，N 512–2048）在旧的固定 tile 梯度下只有约 14 TFLOP/s：28 个 SM 上只有 48 个块。使用调优存储后，它们在 RTX 3060 上的平均时间从 76.5 µs 降到约 60 µs。

## 何时进行测量 {#when-measurement-happens}

算子在被调用的地方、即构建计算图时进行测量，从不在运行中的计划内部测量。每个候选被构建为程序，在按容量分配的临时缓冲区上启动（所有运行时变量都绑定到最大值），并计时：

| 步骤 | 设置 |
|---|---|
| 预热 | 500 ms 的连续运行（RTX 3060 空闲时运行在 210 MHz） |
| 轮次 | 对所有候选进行 4 轮轮询 |
| 每轮 | 10 ms 的维持运行，然后 5 次剖析运行，保留每个候选的最小值 |
| 一次运行的时间 | 最长内核的 GPU 时间戳 |

一个候选是一组程序：对拆分注意力而言是一个内核加上合并其部分结果的内核，其时间是各程序时间之和。构建或运行失败的候选被跳过。如果什么都没测到，就使用第一个候选，并且不存储任何内容。

## 存储 {#the-store}

| 项目 | 值 |
|---|---|
| 目录 | `$SVOD_TK3_TUNE_DIR`，否则 `$XDG_CACHE_HOME/svod/tk3_tune`，否则 `~/.cache/svod/tk3_tune` |
| 文件 | 每个设备和 crate 版本一个，例如 `sm_86_28sm-v0.2.0.txt` |
| 行 | `op\|device\|dtype\|shape\|candidates\|programs index ns` |
| 键 | `tune::TuneKey { op, device, dtype, shape, candidates }`，其中 `device` 是架构和 SM 数量（`sm_86-28sm`） |

来自真实存储的一行：attention，bf16，形状 `[batch, t, tk, heads, kv_heads, d]`，候选 2 以 61.4 µs 胜出。

```text
attention|sm_86-28sm|BFloat16|1x704x704x8x8x64|3733e8921b15aa79|77aee9d5c98fa463 2 61440
```

`programs` 字段是已构建候选程序及其降级的指纹，因此内核一旦改动就会重新测量。写入时会重新读取、合并并原子地替换文件。无法读取或写入的存储视为未命中，从不报错。进程内的记忆缓存无需构建任何东西即可回答重复调用。

## 关闭调优 {#switching-it-off}

| 方式 | 作用 |
|---|---|
| `SVOD_TK3_TUNE=0` | 关闭测量；运行第一个候选 |
| `svod_tk3::tune::set_enabled(false)` | 对当前进程效果相同，优先于环境变量（测试中使用） |

`tune::TuneStore::at(root)` 在另一个根目录构建存储，传入 `None` 时仅存在于内存中。`tune::measure(candidates)` 以同样方式为任意 `tune::Candidate`（`Vec<(Program, Lowering)>`）列表计时。

## 探针 {#probes}

探针是带 `#[ignore]` 的测试，只打印耗时，从不断言。在空闲的 GPU 上逐个运行：

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk3 --lib --release -- --ignored --nocapture --test-threads=1 gemm_candidates_probe
```

| 探针 | 输出 |
|---|---|
| `gemm_throughput_probe` | tk3 GEMM 各配置在 4096³ 上与 tk1 对比的 TFLOP/s |
| `gemm_candidates_probe` | Nemotron 投影形状和 4096³ 上的每个 GEMM 候选、未调优时的选择以及胜者 |
| `attention_throughput_probe` | Flash attention（B 4、H 8、T 2048；d 64/128，因果与非因果）与 tk1 对比 |
| `decode_throughput_probe` | 一个 Whisper large-v3 解码器步骤的自注意力和交叉注意力与 tk1 对比 |
| `first_execution_probe` | 构建、降级和准备一个 tk3 GEMM 的主机开销，与图 GEMM 对比 |

最后一个探针测得的主机开销决定了 `launch.rs` 的设计。降级一个 GEMM 主体需要 2.3 ms，因此降级后的主体按程序、降级、设备和占位符形状被记忆化缓存（命中仅需 30 µs）。在 prepare 阶段，调度一个主体仍比图 GEMM 多花约 0.35 ms。
