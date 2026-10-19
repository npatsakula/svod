---
sidebar_label: 测试与调试
---

# 测试与调试 {#testing-and-debugging}

每个 tk 内核都在两个层面上检查。在主机上，其 tile 程序在解释器中运行，并与直接参考实现比较；降级后的程序会被渲染，并在安装了工具链的地方为每个目标编译。在有 tk 内核的 GPU（CUDA 或 AMD）上，降级后的内核与解释器或该算子自身的计算图回退比较。测试在容差范围内比较数值，而不是比较 IR 的哈希。

## 解释器 {#the-interpreter}

`interp::run(&program, params, vars)` 在主机上执行 tile 程序。它为每个参数接收一个 `Vec<f64>`，并返回运行后的所有参数。每个块按顺序执行其语句，流水线以其串行交错的方式运行。每个值都会舍入到其元素类型（`interp::round_to`），因此结果就是正确的降级在累加顺序误差之内必须重现的值。标量除法向零截断，与发射出的代码一致。运行时变量按名称绑定：

```rust
let out = run(&prog, vec![a.clone(), b.clone(), vec![0.0; m * n]], &[("b", 1)]).unwrap();
```

它返回 `interp::Error::{UnboundVar, ParamSize, Division, OutOfBounds}`：超出缓冲区容量的访问是一个错误，而不是一次静默读取。它不模拟线程和屏障，因此顺序错误只会在设备上暴露。

## 测试文件 {#test-files}

| 文件（`test/unit/`） | 层面 | 检查内容 |
|---|---|---|
| `atoms.rs`, `layout.rs`, `layouts.rs` | 主机 | 每个原子的操作数布局与厂商 ISA lane 公式的对比、F2 代数定律、推断、重布局与 swizzle |
| `build.rs`, `interp.rs`, `schedule.rs`, `sync.rs`, `emit.rs` | 主机 | 构建器记录、解释器语义、流水线展开、屏障 pass、发射出的列表 |
| `ops_plan.rs` | 主机 | 每个算子在各种形状、数据类型和目标下的 `Plan` 与错误 |
| `offline.rs` | 主机 | 显卡目标，以及从拷贝来的调优存储中读取的代价 |
| `targets.rs` | 主机 + 设备 | 每个目标上的每个内核族：降级、在降级前后各解释一次并编译（`ptxas`、clang）；在 GPU 上运行设备门控 |
| `probe.rs` | 主机 + 设备 | 每个代价探针都能为每个目标构建；在 GPU 上运行 `device_costs_are_physical` |
| `attention.rs`, `heads.rs`, `rows.rs`, `conv.rs`, `gather.rs` | 主机 + 设备 | 程序与 f64 参考比较，然后降级后的内核与程序比较 |
| `device.rs`, `parts.rs` | 设备 | 端到端的 GEMM，以及注意力所组合的各部件（在 `cp.async` 设备上） |
| `ops.rs` | 设备 | 每个算子在刁钻形状上与其计算图回退比较 |
| `tune.rs` | 主机 + 设备 | 存储往返、键稳定性、一次真实的 GEMM 调优 |

## 运行测试套件 {#running-the-suite}

`SVOD_DEVICE` 选择设备。只有在无处可运行时，设备测试才会输出一行 `skipped: …` 并跳过：`SVOD_DEVICE=CPU`、tk 没有内核的 GPU，或者既没有设置 `SVOD_DEVICE` 也没有连接受支持的 GPU。如果连接了受支持的 GPU 却没有设置 `SVOD_DEVICE`，它们会失败并指出应设置的值，因为设备测试全部跳过的套件虽然能通过，却什么也证明不了。

```bash
# Host only, no GPU:
SVOD_DEVICE=CPU cargo test -p svod-tk --release --lib
# On the GPU; on a fresh tune store run the cost probes alone first, since the parallel
# suite inflates what they measure and the store keeps the first measurement:
SVOD_DEVICE=CUDA cargo test -p svod-tk --release --lib device_costs_are_physical
SVOD_DEVICE=CUDA cargo test -p svod-tk --release --lib
```

`SVOD_DEVICE=AMD`（或 `AMD:N`，按 KFD 节点顺序）选择 AMD GPU。只有当 `families_match_the_interpreter_on_the_device` 在某个目标上通过后，该目标才算已测量：它把每个内核族的第一个候选在设备上的结果与解释器比较。`register_staged_families_match_on_cuda` 在 CUDA GPU 上运行 AMD 的数据路径（寄存器暂存填充，不使用 `ldmatrix`）。

### 在新的 GPU 上 {#on-a-new-gpu}

1. 把调优存储指向一个要保留的目录：`SVOD_TK_TUNE_DIR=$PWD/tk_tune_$(hostname)`。
2. 在有多个 AMD 节点的机器上，检查 `SVOD_DEVICE=AMD:N` 所指的节点确实是该 GPU（`/sys/class/kfd/kfd/topology/nodes/*/properties`，`gfx_target_version`）。
3. 单独运行 `device_costs_are_physical`，然后运行 `families_match_the_interpreter_on_the_device`，再运行整个套件。不要报告门控未通过的设备的数字。

Nemotron 的模型级门控（真实权重、生成的金标准）：

```bash
SVOD_DEVICE=CUDA cargo test -p svod-model --release --lib nemotron_diar::parity::half_precision -- --ignored --nocapture --test-threads=1
```

## 内核运行了吗？ {#did-the-kernel-run}

发生回退的算子构建的是计算图算子而不是 `CALL`；`ops.rs` 的测试在结果计算图的调用中查找内核名称。要在没有设备的情况下预测走哪条路径，可以调用 `ops::shape` 规划器（参见[算子层](./op-layer#kernel-or-graph)）；在设备上，规划器日志会说明某个算子为何走了计算图（[调优](./tuning#the-planner-log)）。

## 为另一张显卡做规划 {#planning-for-another-card}

`tk_dump` 示例展示 tk 在本机没有对应设备的显卡上会运行什么：规划器的排序及其预测的 ns，以及格点中每个配置被略去的原因（`plan.txt`）；所选配置的 LLVM 文本，以及在安装了相应工具的地方，其 PTX、`ptxas -v` 资源和 SASS（CUDA）或带资源注释的 AMDGCN 汇编（AMD）；还有主循环的操作码直方图。

```bash
cargo run -p svod-tk --release --example tk_dump -- --card h100-pcie --op gemm --shape 4096x4096x4096
cargo run -p svod-tk --release --example tk_dump -- --arch gfx942 --op attention --shape 8x16x1024x128 --causal --store DIR
```

`--card` 接受 `atoms::CARDS` 中的名称（`rtx3060`、`h100-pcie`、`h100-sxm`、`rtx5070`、`rtx5090`、`mi300x`、`mi355x`、`rx9070xt`、`strix-halo`）；`--arch` 接受一个架构，配合 `--sms`/`--xcds` 使用，否则取列出的第一张卡的值。`--store` 是从该卡拷贝来的调优存储目录，其探针行为排序定价；没有它时则用猜测的代价。`--cfg N` 发射排序中第 N 个配置，`--out DIR` 指定输出目录（`tk_dump`）。

`--op` 接受 `gemm`（`--shape MxNxK`）、`attention`（`BxHxTxD`）、`norm`（`RxD`）、`conv`（`BxHxWxCINxCOUT`）或 `heads`（`BxTxHxKVxD`），每一种都经过算子层自己的形状关卡和调优。`--cfg tuned` 发射调优器在该卡上选中的配置，从 `--store` 副本中读取且不向其写入任何内容：存储的行按算子、设备、数据类型、形状以及算子层作为键的配置列表匹配，不论写入它的是哪个编译器或哪次内核构建；转储会打印读取的每个键，或指出缺少哪一行。`--compare FILE [--fun NAME]` 把参考内核（`.cubin`、AMD 代码对象，或 SASS、PTX、AMDGCN 文本）的主循环与 tk 的主循环并排打印，二者按同一方式计数。

```bash
cargo run -p svod-tk --release --example tk_dump -- --card rtx3060 --op gemm --shape 4096x4096x4096 --cfg tuned --store ~/.cache/svod/tk_tune
cargo run -p svod-tk --release --example tk_dump -- --card mi300x --op gemm --shape 4096x4096x4096 --compare hk.co --fun NAME
```

## 环境变量 {#environment-variables}

| 变量 | 作用 |
|---|---|
| `SVOD_DEVICE` | 默认设备：`CUDA[:N]`、`AMD[:N]`、`CPU`。它决定内核是否适用，以及运行哪些设备测试 |
| `SVOD_TK_TUNE=0` | 不对候选计时；运行第一个干净的配置（参见[调优](./tuning)） |
| `SVOD_TK_TUNE_DIR` | 调优存储的目录 |
| `RUST_LOG=svod_tk::plan=debug` | 规划器的决策；`svod_tk=debug` 还会加上调优器的各轮计时 |
| `TK_DUMP_LIST=1` | 程序降级时打印每条发射的指令（`[i] id op dtype <- sources`）。主体会被记忆化缓存，因此只打印每个程序的第一次降级 |
| `SVOD_SPEC_DEBUG=1` | 程序未通过 IR spec 校验时，打印被拒绝的指令及其树 |

## 剖析模型 {#profiling-a-model}

Nemotron 示例先预热一次（同时填充调优存储），然后对第二次运行计时并剖析，打印其逐内核报告：

```bash
cargo run -p svod-model --release --example nemotron_diarize -- audio_1.wav --dtype bf16 --profile
```

tk 内核以其内核名称（`gemm`、`conv`、`flash_attention`、`heads`、`layer_norm`……）与图内核并列出现，并以相同方式计时。

:::tip[在安静的 GPU 上测量]
GPU 上的其他进程会使每个数字产生偏差。探针和调优存储会先预热时钟。运行探针时使用 `--test-threads=1`。
:::
