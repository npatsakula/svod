---
sidebar_label: 剖析与基准测试
---

# 剖析与基准测试内核

[调试](./debugging) 用单次硬件时间戳回答「这个内核对不对、大致有多快」。本章则讲它之后的那个问题：*时间都花到哪去了，瓶颈在哪里？* Svod 附带一个**分层内核 profiler**，分四个层级来回答它——设备时间、roofline、静态占用率，以及硬件计数器——全都统一到一次调用之后。

这个 profiler 位于 `runtime` crate 而非 `tk`，这一安放位置正是关键所在：它对**任何** `Tensor` 或 `ExecutionPlan` 都管用，无论其中的内核出自图优化器，还是由 `tk` 手写而成。一个图 matmul、一个融合的前馈块、一个手写的 Flash Attention，全都出现在同一张表里，以同样的方式计时、分析。之所以放在本节介绍，只是因为手写内核的作者最可能用到它。

:::note[框架级，而非仅限 tk]
下文的一切都适用于任何可实现的 `Tensor`。[一套 IR 的设计](./lowering) 正是这一切的前提：一个 `tk` 内核不过是更多的 UOp，因此它把自己的 `name` 一路带进设备 profile，并由与自动调优内核完全相同的路径来测量。
:::

---

## 四个层级

每个层级都往报告里添上一组列。较低的层级开销小、随时可用；较高的层级则需要更多前提（一份估算、一次描述符解码、一块状态稳定的 GPU）。开销大的那些要你主动开启。

```mermaid
flowchart TD
  P["Tensor::profile / ExecutionPlan::profile"] --> T1["Tier 1 - device time (GPU-clock timestamps)"]
  P --> T2["Tier 2 - roofline (GFLOP/s, GB/s)"]
  P --> T3["Tier 3 - static occupancy (VGPR/SGPR/LDS, occ%)"]
  P --> T4["Tier 4 - HW counters / PMC (AMD SQ block, CUDA CUPTI)"]
```

| 层级 | 报告什么 | 来源 | 需要执行吗？ |
|------|-----------------|--------|------------------|
| **1 — 设备时间** | 每个内核的 GPU 执行时间 | GPU 时钟派发时间戳 | 是 |
| **2 — roofline** | 推导出的 **GFLOP/s** 与 **GB/s** | FLOP 由内核的 IR 估算；字节数来自 plan 的缓冲区 | 是（速率需要时间） |
| **3 — 静态占用率** | VGPR / SGPR / LDS / scratch 用量，以及**占用率 %** | AMD：从内核描述符解码而来。CUDA：`cuFuncGetAttribute` 加上 `cuOccupancyMaxActiveBlocksPerMultiprocessor` | 无需 dispatch——AMD 上是一次静态解码，CUDA 上是一次驱动查询 |
| **4 — 硬件计数器（PMC）** | AMD：SQ 忙碌周期、wave 数、VALU 指令数。CUDA：SM 周期、warp 数、指令数、tensor 流水线周期、DRAM 字节数 | AMD：PM4 性能计数器包跨网格求和。CUDA：CUPTI range profiler | 是，且计数器必须已解锁 |

有几个细节值得了解：

- **第 2 层**通过遍历内核的 IR（AST）来估算 FLOP。对调度器构建的内核而言，其范围是有界的，于是估算就是一个真实计数，GFLOP/s 这一列也就有了值。GB/s 这个数字把每个不同的 LOAD/STORE 缓冲区各算一次，所以只要第 2 层在跑，它就可用。
- **第 3 层**在 AMD 上直接从内核描述符解码各项资源（VGPR/SGPR/LDS/scratch），完全不需要访问 GPU。它的占用率 % 只为 gfx11（RDNA3/3.5，wave32）建模，因为那里寄存器堆的几何结构是已知的；在 CDNA3 (wave64) 上，各项资源仍会显示，但占用率这一列会显示 `-`。AMD 上的这个数字仅是**受 VGPR 限制**的一阶限制因素——LDS 与工作组上限并未折算进来。在 CUDA 上，这些数字改为取自已加载的函数：每线程寄存器数、静态共享内存和 local（scratch）字节数来自 `cuFuncGetAttribute`，占用率则由*驱动*算出（对最近一次启动的块大小调用 `cuOccupancyMaxActiveBlocksPerMultiprocessor`，再除以 SM 的线程容量），因此它确实把共享内存和块形状折算了进来。CUDA 上没有 SGPR 这一列。
- **第 4 层**因后端而异。在 AMD 上它用 PM4 包对 SQ 块编程并跨网格求和：`sqbusy`（忙碌周期）、`waves`（启动的 wave 数）和 `valu`（发射的 VALU 指令数）——它们合在一起回答的，是单凭计时无法回答的 ILP / 占用率问题。在 CUDA 上它驱动 CUPTI range profiler：`cycles`、`warps`、`inst`、`tensor`（tensor 流水线活跃周期）和 `dram`（经过 DRAM 的字节数）——`tensor` 比上 `cycles` 就是 tensor core 利用率，而 `dram` 能把受带宽限制的内核和受发射限制的内核区分开。计数器 token 在各后端之间唯一，所以一份写了别的后端计数器的选择只会把它们丢掉，而不会去错误地编程另一个模块。

报告的各列会随采集到的内容自适应：若只跑了第 1 层，便仅打印计时；GFLOP/s、资源和计数器各列只有在对应层级运行过后才出现。

---

## API：在 `Tensor` 或 `ExecutionPlan` 上的 `profile`

有两个入口点。两者都接收一个 `&ProfileOptions` 并返回一个 `RunProfile`。

```rust
// tensor/src/realize.rs — realizes the tensor as a side effect, like realize()
pub fn profile(&self, opts: &ProfileOptions) -> Result<RunProfile>

// runtime/src/execution_plan.rs — profile an already-prepared plan
pub fn profile(&self, opts: &ProfileOptions) -> Result<RunProfile>
```

`Tensor::profile` 是方便的那个：它会准备好 plan，跑一遍剖析路径，再敲定结果，使张量最终的实现状态与 `realize()` 所留下的完全一致。`ExecutionPlan::profile` 则适用于你已经握有现成 plan 的场合（基准和 `Tensor::profile` 底层调用的都是它）。

```rust
use svod_runtime::ProfileOptions;

// Any Tensor — a tk kernel here, but a pure graph computation works identically.
let out = svod_tk::flash_attention(&q, &k, &v)?;
let report = out.profile(&ProfileOptions::default())?;

// The library NEVER prints. render_table() returns a String; the caller decides.
print!("{}", report.render_table());
```

或者针对一个你自己准备好的 plan：

```rust
let plan = out.prepare()?;
let report = plan.profile(&ProfileOptions::from_env())?;
print!("{}", report.render_table());
```

`RunProfile::render_table()` 返回一个 `String`——一张按内核排列的表（内核按入口点聚合，按总时间排序），并带上所有有值的层级列。这个 profiler 是个纯格式化器：它自己从不写 stdout 或 stderr，所以日志、文件和 stderr 回显始终由调用者决定。

---

## `ProfileOptions` 与 `from_env`

```rust
// runtime/src/profiler.rs
pub struct ProfileOptions {
    pub iters: u32,             // replays; the per-kernel minimum device time is kept
    pub static_analysis: bool,  // Tier 2/3 (flops/bytes/resources) — cheap, on by default
    pub counters: PmcSelection, // Tier 4 hardware counters
    pub origin_depth: Option<usize>, // origin rollup depth; None keeps the full path
}
```

`ProfileOptions::default()` 即 `{ iters: 1, static_analysis: true, counters: PmcSelection::None, origin_depth: None }`——第 1–3 层，单趟。想要显式控制就直接构造它：

```rust
use svod_runtime::{ProfileOptions, PmcSelection};

let opts = ProfileOptions {
    iters: 50,
    static_analysis: true,
    counters: PmcSelection::Default, // add Tier 4
    origin_depth: Some(3), // roll the origin rows up to three frames
};
```

`PmcSelection` 可取 `None`（仅第 1–3 层）、`Default`（当前后端所采集的那一组，经由 `PlanContext::pmc_default` 解析），或 `Custom(Vec<PmcCounter>)`（一份显式列表）。

`ProfileOptions::from_env()` 是读取剖析环境变量的唯一地方：

| 环境变量 | 作用 |
|---------|--------|
| `SVOD_PROFILE_ITERS` | 用于取最小合并的重放次数（钳制为至少 1） |
| `SVOD_PMC` | 第 4 层的选择：空或 `0` → 关闭；`1` → 该后端的默认计数器组；否则为一份逗号分隔的 token 列表（AMD 为 `sqbusy`、`waves`、`valu`；CUDA 为 `cycles`、`warps`、`inst`、`tensor`、`dram`） |
| `SVOD_ORIGIN` | 除空值和 `0` 以外的任何取值都会记录每个操作构建时所处的作用域（模块路径、调用点、ONNX 节点），见下文——它是在 `svod-ir` 里操作构建时读取的，并非由 `from_env` 读取 |
| `SVOD_ORIGIN_DEPTH` | 来源汇总保留的路径段数（`origin_depth`）；未设置或为 `0` 时保留完整路径 |

```bash
# Profile with 20 replays and the default hardware counters.
SVOD_DEVICE=AMD:0 SVOD_PROFILE_ITERS=20 SVOD_PMC=1 ...

# Only VALU instructions and SQ-busy cycles.
SVOD_DEVICE=AMD:0 SVOD_PMC=valu,sqbusy ...

# Tensor-core utilization and DRAM traffic on CUDA.
SVOD_DEVICE=CUDA:0 SVOD_PMC=tensor,dram ...
```

### 累积取最小

当 `iters > 1` 时（或跨 criterion 的多次调用），profiler **不会**取平均。每一趟都产出一个 `RunProfile`，各趟由 `RunProfile::merge_min` 合并：对每个内核，更快（设备时间最小）的那个样本胜出，并带上*那个*样本的静态分析。最小值是内核内在开销的稳健估计量——它把调度抖动、争用以及时钟爬升这些会抬高均值的离群点都剔除掉。

计数器是个例外：采集计数器会扰动采集它的那一趟，所以那一趟永远不会是最快的；合并时会保留真正采到计数器的那一趟的计数器，而不是把它们连同较慢的样本一起丢掉。因此同一张表里的计时和计数器可能来自不同的趟——这正是本意，因为带计数器的那一趟并不能用来给内核计时。

## 把内核归属到模型代码 {#attributing-kernels-to-model-code}

`r_128_3_32_4_2_2_2_4_4_192_2` 这样的内核名只说明内核的形状，不说明它服务于哪一层。开启 `SVOD_ORIGIN=1` 后，每个张量操作都会记下自己构建时所处的作用域——`encoder.layers.3.ffn1` 这样的模块路径、公开操作的调用点，或是 ONNX 节点索引——调度器再把这些来源的并集带到每次 dispatch 上。十六个完全相同的层仍然只编译出一个程序，只是 dispatch 十六次，各自带上一份归属。

模型沿着 state-dict 路径打开作用域（`OriginScope::module`），ONNX 导入器为每个节点打开一个，阶段名（`vad`、`encoder`、`ctc_head`）则是根部的标签。手写的 `tk` 内核适用同一条规则：构建它时处于活动状态的作用域就是它的来源。

一次运行带有来源时，`render_table()` 会附加两份汇总：

- **exclusive** 把每次 dispatch 只计一次，记到它的主要来源（产生所存储值的那个作用域）上，因此各行加起来正好是总量；
- **inclusive** 把每次 dispatch 记到融合进它的每个来源的所有祖先上，因此父行包含子行，各行互相重叠。

两者都截断到 `origin_depth` 段；调用帧（`@ add tensor/src/arithmetic.rs:31`）只作为细节留在内核行里，绝不构成汇总键。在任何作用域之外构建的内核落到 `<unattributed>` 行。深度记录在 `RunProfile` 上，因此 `render_table()`、`Display` 和 `to_json()` 都按该 profile 产出时的深度截断（`SVOD_ORIGIN_DEPTH` 也算在内）；要另选深度就用 `render_table_at(d)` / `to_json_at(d)`。

```
origin rollup (depth 3, exclusive; rows sum to the total):
  total ms  count    mean µs      %  origin path
    23.045      2    11522.6    5.3  ctc_head.GigaAmCtcJit.subsampling
     8.231      3     2743.7    1.9  ctc_head.GigaAmCtcJit.layers.6
```

`RunProfile::to_json()` 导出三部分：内核行（每行在渲染好的路径旁还带上原始的 `origin_id` / `origin_ids`）、两份汇总，以及这些 id 能触及的那些 arena 条目，形如 `{ id, parent, frame }`——这样既能离线还原路径，又不必把整个进程级 arena 塞进文件。`gigaam_infer --profile-json out.json --origin-depth 3` 就会写出这样一个文件。

开启捕获会改变节点身份：不同作用域下构建的两个相同子图，在内核切分之前不再合并。内核程序不受影响，但那种在每个调用点重建同一表达式的辅助函数应放在 `OriginScope::suspend()` 下运行，或者直接给它传入已经物化好的输入。

---

## Criterion 集成：`--profile-time`

`tk` 的基准通过每个内核公开的 `Tensor` 接口来测量内核，计时用的是 profiler 所用的那同一批每内核 GPU 时间戳（`tk/benches/common.rs`）。普通的 `cargo bench` 只报告每个基准的 GPU 设备时间。但 criterion 有一个 `--profile-time <seconds>` 模式，而这些基准通过 criterion 的自定义 `Profiler` trait 把**完整的分层 profiler** 挂接了进去——这正是火焰图生成所用的那同一个扩展点。

这个挂钩就是 `tk/benches/common.rs` 里的 `PlanProfiler`。在剖析某个基准期间，`bench_plan`（或 `bench_kernel`——它只统计入口点带有手写内核名字的那些 dispatch，因此实现单个输出时额外加上的那次拷贝不会算到内核头上）会在每次调用时通过进程全局的 `bench_profiler()` 捕获该基准的 plan，每次捕获都经 `ProfileOptions::from_env()` 剖析，并按每内核最小值合并进会话累加器。停止时，合并后的表用 `render_table()` 渲染，写入 criterion 输出目录下的一个文件，并回显到 stderr：

```
target/criterion/<id>/profile/svod-profile.txt
```

接入它只需在每个基准的 `criterion_group!` 里加一行——这一行把共享 profiler 设为 criterion 的配置（取自 `tk/benches/fa.rs`）：

```rust
criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(common::bench_profiler());
    targets = bench_fa
}
criterion_main!(benches);
```

像跑任何 criterion 基准那样运行它，再加上 `--profile-time`（以及任何层级环境变量）。每个内核一个二进制：`fa`、`gemm`、`matmul`、`norm`、`sq_attention`、`knn`、`kmeans`。

```bash
# Plain bench: GPU device time per benchmark, profiler dormant.
SVOD_DEVICE=AMD:0 cargo bench -p svod-tk --bench fa

# Drive the layered profiler for ~5s per benchmark, with hardware counters.
SVOD_DEVICE=AMD:0 SVOD_PMC=1 cargo bench -p svod-tk --bench fa -- --profile-time 5
```

由于 `bench_profiler()` 在 criterion 不剖析时处于休眠，普通的 `cargo bench` 完全不受影响——还是同样的数字，没有额外的趟数。在内核 `ArchSet` 之外的设备上，基准会自行跳过。注意，除非设置 `SVOD_TK_TUNE=0`，基准中的 [自动调优器](./tuning) 是*开启*的：新形状的第一个样本包含分块测量的时间，这部分会被 criterion 的预热吸收掉。

---

## 如实交代局限

:::caution[profiler 给不了你的两样东西]
**第 2 层的 GFLOP/s 对手写内核是空白。** FLOP 估算会遍历内核的 IR，而它只会为**调度器构建的**内核自动计算速率。它是从一个操作的操作数依赖什么，来推断该操作位于哪些循环里的——只要索引表达式由调度器来写，这就成立。而手工 lower 的 `tk` 内核自己做寻址，它的循环变量只经由地址抵达算术，于是这趟遍历就再也还原不出嵌套关系了，两个方向都不行。profiler 会拒绝给出估算，而不是打印一份垃圾 roofline（早期版本曾把一个 matmul 报成硬件峰值的几十倍），所以对那些内核**GFLOP/s 这一列显示 `-`**。（GB/s 仍然有效，因为字节数来自 plan 的缓冲区，而非 IR。）手写内核的 roofline 请你自己来算：用算法已知的 FLOP 计数和第 1 层的设备时间。

**第 4 层必须先解锁，而且各家厂商的要求不同。** 在 AMD 上，PM4 硬件计数器只有在 GPU 保持固定时钟时才有意义，所以 GPU 必须处于 `profile_standard` 电源状态（`amd-smi set -l stable_std`）。在 CUDA 上，除非设置了 `NVreg_RestrictProfilingToAdminUsers=0`，驱动只允许管理员采集计数器，并且 CUPTI 必须能被加载（`SVOD_CUDA_CUPTI=0` 可以主动关掉它）。两种情况下 profiler 都*不会*失败：它只报告计时，并打印一行提示说明缺了什么。NVIDIA 的细节见[在 CUDA 上剖析](../backends/cuda/profiling.md)，其中也讲了为什么那里采集计数器要多花一趟。
:::

---

## 哪个问题用哪个调用

| 你在问…… | 用 |
|----------------|-----|
| 「每个内核在这块 GPU 上要花多久？」 | `Tensor::profile` 配 `ProfileOptions::default()`，读设备时间那一列 |
| 「这个内核是计算受限还是带宽受限？」 | 第 2 层的 GFLOP/s 与 GB/s 两列（图内核），或手工算 roofline（tk 内核） |
| 「占用率为什么低——是寄存器还是 LDS？」 | 第 3 层的 VGPR/SGPR/LDS/占用率 % 各列（无需计时运行） |
| 「内核每个忙碌周期发射的 VALU 工作量够吗？」 | 第 4 层 `SVOD_PMC=1`，在 `profile_standard` 状态的 GPU 上 |
| 「内核到底有没有用上 tensor core，还是卡在 DRAM 上？」 | CUDA 上的第 4 层 `SVOD_PMC=tensor,dram` |
| 「跨多次运行，它与图原生基线相比如何？」 | `cargo bench --profile-time`——见 [调试 → 在真实硬件上计时](./debugging) |

若要做的是正确性与结构检查而非性能，请留在 [调试](./debugging)；至于内核*之下*的问题（队列、故障、驱动），见 [AMD 后端 → 调试](../backends/amd/debugging) 或 [CUDA 后端 → 调试](../backends/cuda/debugging)。
