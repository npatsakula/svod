---
sidebar_label: 剖析
---

# 用线程追踪剖析 {#profiling-with-a-thread-trace}

内核的时间告诉你它慢。线程追踪告诉你为什么慢：每个 wave 在等待哪条指令、等了多久，以及这条指令来自 tile 程序的哪一行。本页带你从“这个内核很慢”走到对 tk 程序的一处改动。

## 为什么需要追踪 {#why-a-trace}

其他工具都从外部观察内核。设备时间说明它耗时多久，roofline 说明它距离内存与计算上限有多远，计数器说明各单元有多忙。它们都说不出某个周期上一个 wave 在做什么。在 AMD 的 RDNA 显卡上，SQ 线程追踪（SQTT）恰好记录了这一点，因此你可以看出主循环是在等待全局加载、等待 LDS、等待屏障，还是什么都没在等。

Svod 的剖析器分为五层。每一层都在下一层之上增加细节：

| 层级 | 提供什么 | 适用范围 |
|---|---|---|
| 1，设备时间 | 每个内核在 GPU 时钟上的时间 | GPU 后端；CPU 上为墙钟时间 |
| 2，roofline | 由第 1 层和内核的静态工作量估计得到的 GFLOP/s 与 GB/s | 所有后端 |
| 3，静态占用率 | 寄存器、LDS、scratch 以及占用率估计，无需派发 | AMD、CUDA；Metal 部分支持 |
| 4，硬件计数器 | 繁忙周期、wave、指令（`SVOD_PMC`） | AMD（PM4）、CUDA（CUPTI） |
| 5，线程追踪 | 每个被追踪 wave 的指令及其周期，只需额外一遍运行 | AMD gfx11 和 gfx12 |

## 追踪记录了什么 {#what-the-trace-records}

追踪是每个着色器引擎（SE）的一条令牌流：一个 wave 何时开始、何时结束，以及对被追踪的 SIMD 而言，它发出的每条指令及发出时间。Svod 解码这条流，把每条指令映射回内核的反汇编，并把主循环的周期划分到各个桶中：

- **issue**：指令正在发出，按其类别区分（`matrix`、`valu`、`salu`、`lds`、`vmem_load`……）；
- **stall**：指令已就绪，但其流水线或队列繁忙；
- **wait**：一条 `s_wait_*` 在等待加载、LDS 访问或存储，计入约束它的那次加载；
- **barrier**：wave 停在工作组屏障处；
- **idle**：没有记录，因为该 SIMD 正在为另一个 wave 发出指令，或在等待追踪看不到的依赖。

每个桶还有一个**暴露**（exposed）份额：其周期减去其他流水线的工作在同一时间掩盖掉的部分。一次 180 个周期的加载等待，若其中 140 个周期矩阵流水线都在忙，则代价约为 40。各项发现按暴露份额排序，因为只有暴露的周期在被消除时才会缩短内核时间。

一次**行程**（trip）是 tile 程序循环体的一次执行；循环展开时，它可能短于一次硬件迭代。周期数指一个被追踪 wave 的周期，按每次行程计。

## 范围与开销 {#scope-and-cost}

- **硬件。** AMD gfx11（RDNA3、RDNA3.5）和 gfx12（RDNA4）。其他任何目标都会以 `only gfx11 and gfx12 are traced` 被拒绝。采集已在 gfx1201（RX 9070 XT）和 gfx1151（Strix Halo）上验证。gfx1100 的追踪可以解码，但尚未验证任何硬件采集。
- **gfx11 没有上限数据。** gfx11 把矩阵操作记录为一个单周期令牌，因此在那里各流水线的繁忙份额未知。报告不打印 `ceiling` 行，其发现使用原始份额，并标记为 `approx: untimed_matrix`。
- **仅限 PM4 队列。** 以 `SVOD_AMD_AQL=1` 打开的设备拒绝启用追踪。
- **窗口。** 每个 SE 默认获得 64 MiB 缓冲区（`SqttConfig::window`，以 4 KiB 页计，可配置）。写满的缓冲区会丢失其尾部。此时报告以 `!! TRUNCATED` 开头，其总数偏少。
- **每个被追踪的内核额外一遍运行。** 剖析先在不追踪的情况下为各次运行计时，然后在启用追踪的情况下再运行一次计划。这一遍在每个内核之后同步，并在下一个内核之前释放其缓冲区。任何计时列都不来自这一遍。
- **两个 SIMD。** 指令令牌来自 SE0 和 SE1，各一个 SIMD（`SqttConfig::se_mask`）。`SVOD_SQTT_SIMD` 选择它们：为每个 SE 指定同一个 SIMD，或给出一个在各 SE 上循环使用的逗号列表（默认 `0,1`，即 SE0 上为 SIMD 0，SE1 上为 SIMD 1）。逐指令的数字描述的是这两个 SIMD 上的 wave。

## 时钟与前提条件 {#clocks-and-prerequisites}

- 一块属于上述目标的 AMD 显卡，以 `SVOD_DEVICE=AMD`（或 `AMD:N`）选中，运行 PM4 队列。
- 与 Svod 编译所用的 clang 放在一起的 `llvm-objdump`。采集会保存其反汇编，因此重放只在缺少 `kernel.s` 时才需要该工具。
- **不固定时钟，这是默认行为。** 追踪的那一遍以设备自行选择的时钟运行，与未追踪的运行一样。`SVOD_SQTT_PSTATE=standard`（或 `peak`）通过 DRM 上下文固定一个稳定的功耗状态。这会降低着色器时钟并扭曲追踪所呈现的内容。仅在需要与旧的固定时钟采集对照时使用。
- 一块安静的 GPU。追踪无论如何都会扰动运行：报告会打印追踪运行时间与未追踪运行时间之比，超过 1.2× 时将发现标记为 `approx: perturbed`。切勿将追踪下的时间与未追踪的时间比较。

## 在 Rust 中追踪 {#tracing-from-rust}

`ProfileOptions::sqtt` 为任何被剖析的计划（`ExecutionPlan::profile`）或张量（`Tensor::profile`）开启第 5 层：

```rust
use svod_runtime::{ProfileOptions, SqttOptions};

let opts = ProfileOptions { sqtt: Some(SqttOptions::default()), ..Default::default() };
let profile = plan.profile(&opts)?;
for traced in profile.stages.iter().flat_map(|stage| &stage.sqtt) {
    match &traced.analysis {
        Ok(analysis) => println!("{}: {}", traced.kernel, analysis.summary()),
        Err(error) => println!("{}: {error}", traced.kernel),
    }
}
```

`SqttOptions` 包含一个 `SqttConfig`（SE、SIMD、窗口、功耗状态）和一个 `Sink`：

- `Sink::Disk(dir)` 为默认值，把每个内核的采集保存在 `dir/<index>_<entry point>` 下（默认 `./sqtt`），因此这一遍同一时间只持有一份采集。`Traced::raw` 为 `Raw::Saved(path)`。
- `Sink::Memory` 把每份采集保存在 `Raw::Held` 中。一份采集每个 SE 可占用多达 64 MiB，因此整个模型的采集可能放不下。

每个 `Traced` 携带自己的 `analysis`。一个内核的追踪无法分析时，其他内核不受影响。`RunProfile::render_report` 以每个被追踪内核一行摘要结尾。

## 追踪模型 {#tracing-a-model}

模型示例的 `--profile` 标志不会追踪。要追踪模型，请用上述选项剖析其输出张量。计划中的每个已编译内核都会被追踪：tk 内核和图内核都一样。`SqttOptions::from_env()` 从环境构建这些选项，供需要开关的程序使用：`SVOD_SQTT=1` 开启追踪，`SVOD_SQTT_DIR` 指定目录，`SVOD_SQTT_PSTATE` 和 `SVOD_SQTT_SIMD` 设置配置。`SVOD_SQTT` 不为 `1` 时它返回 `Ok(None)`，值无法解析时返回错误。

从未追踪运行的逐内核表开始，挑出占主导的内核，用 `tk_bench sqtt --load`（见下文）读取它的目录。对于 tk 内核，接下来单独追踪它，以便改变其配置。

## 追踪单个内核 {#tracing-one-kernel}

`tk_bench sqtt` 在某个形状上构建一个 tk 算子，像调优器那样在不追踪的情况下为每个配置计时，追踪一次，保存采集并打印其报告：

```bash
SVOD_DEVICE=AMD cargo run --release -p svod-tk --example tk_bench -- sqtt nearest 4096 128 16
```

| 命令 | 追踪对象 |
|---|---|
| `sqtt gemm M N K [--residual] [--gated]` | bf16 GEMM，带残差或 SwiGLU 尾处理 |
| `sqtt attention B H T D [--causal] [--bias] [--flip FORM,…] [--ring S]` | `[B, T, H, D]` 上的 bf16 注意力；`--flip` 翻转每个配置的 softmax 形式，`--ring` 设置其环形缓冲槽位数 |
| `sqtt nearest N D K [--metric M] [--splits S]` | `N` 行深度为 `D` 的 f16 数据的自连接，每行保留 `K` 个；`--splits` 覆盖配置的拆分数 |
| `sqtt conv H W CIN COUT KERNEL STRIDE [--batch B] [--residual]` | 一个 YOLO 块的 f16 卷积，带偏置和 SiLU |

四个命令都接受 `--cfg`、`--top N`（表格行数，默认 12）、`--out DIR`（默认 `./sqtt`）和 `--json`（报告输出为一行 JSON）。`--cfg` 是逗号列表。默认值 `tuned` 是该算子会启动的配置。数字 `I` 是规划器排名第 `I` 的配置。GEMM 还接受固定编码 `BMxBNxBK:S:WRxWC[:rolled]`，`nearest` 和 `conv` 接受 `ranked`，它列出排名而不进行追踪。每个配置落在 `DIR/<op>_<shape>_<cfg>/p<program>/<index>_<entry>/` 中，路径以 `saved …` 打印。

## 保存的目录与重放 {#the-saved-directory-and-replay}

| 文件 | 内容 |
|---|---|
| `se0.att`、`se1.att`、… | 每个 SE 的原始追踪流 |
| `kernel.co` | 被追踪的代码对象；在 AMD 上启动的 tk 内核还在其 DWARF 中携带针对其 `kernel.tk` 清单的行号表以及该清单本身 |
| `kernel.s` | `kernel.co` 的 `llvm-objdump -d` 输出；报告中的 PC 即其地址 |
| `meta.json` | `format` 2、架构、加载基址、未追踪与追踪的 ns、各 SE、标签（算子、形状、配置）、循环模型、资源、工作组与网格、功耗状态、显存时钟 |

加载时会检查 `format`，不一致时要求你重新采集。每次加载都会从 `kernel.co` 读回语句。只有 AMD 的 tk 内核有语句；图内核没有（CUDA 和 Metal 没有线程追踪）。没有语句时，一次行程被视为一次循环迭代，并标记为 `approx: per_iteration`。

重放不需要 GPU：

```bash
cargo run --release -p svod-tk --example tk_bench -- sqtt --load DIR [--json] [--attribute]
```

在 Rust 中，`svod_runtime::sqtt::Analysis::load(dir)` 返回同样的分析结果。

## 阅读报告 {#reading-the-report}

`rdna4_nearest_split` 夹具（`device/src/test/fixtures/sqtt/svod/`）是一个 gfx1201 上的 nearest 内核，n4096 d128 k16。节选后，其报告如下：

```text
== verdict: cycles are a traced wave's, a trip (1 an iteration, its statements' unroll copies); exposed: not hidden by other pipes' work
ceiling    matrix pipe 14.7% busy over the steady window: removing every stall gains at most 6.82x; …
bound      occupancy_limited  56.6%  SIMD idle 56.6%: 2.0 waves a SIMD, ~4.6 would fill it; limit 8 by vgpr+lds
   knob: fewer waves resident than the kernel allows: more workgroups in flight
 1 barrier            18.6% (raw 23.4%)  barrier (sink): 255/trip parked for the workgroup's slowest wave, 203 exposed; …
   knob: drop or merge a barrier only if its exposed share passes 10%
 2 load_exposed       13.0% (raw 16.8%)  cycles a trip: waiting on loads 183 (141 exposed); stalled issuing 0 (0 exposed)
   0x364 load exposed 129/trip at wait 0x44c (then ds_store_b128 v177, v[105:108]): cover med 91, latency med 238, … @21 fill c staged issue [steady]
   knob: latency: more work between a load and its wait (prefetch distance, ring slots)
 3 loop_overhead       2.7% (raw 3.2%)  valu 26 + salu 9 issue cycles a trip beside matrix 146 (24%), …
kernel     144840 ns untraced, 148360 ns traced (1.02x), wave span 461319 cycles, sclk ~3109 MHz, mclk 875 MHz, pstate unpinned
traced     se0 sa0 wgp0 simd0 (2 waves), se1 sa0 wgp0 simd1 (2 waves) of 2 SEs
== statements: main loop cycles a trip by kernel.tk line (a wait on its binding load's); 104 of 105 loop instructions attributed; top 12 by exposed
 line  pcs    issue    stall     wait  barrier     idle    total  exposed  statement
 1581    1      0.0      0.0      0.0    255.1      0.0    255.1    202.6  await [steady]
   21    8      7.0      2.1    182.7      0.0     32.7    224.5    162.5  fill c staged issue [steady]
   28   24    145.7     11.4      0.0      0.0     75.5    232.6    148.0  mma x·c_s [steady]
```

自上而下阅读：

1. **`ceiling`**：矩阵流水线有多忙。在 14.7% 时，这个循环离矩阵上限很远，消除所有停顿至多获得 6.82× 的提升。
2. **`bound`**：内核的类别，由各流水线的繁忙份额得出（`matrix_bound`、`issue_bound` 或 `occupancy_limited`）。
3. **编号的发现**：循环的暴露周期花在了哪里，从大到小排列。每项列出其证据（PC、等待背后的加载、最后到达屏障的是谁）、`@` 之后的 tk 语句，以及说明应改动什么的 `knob` 行。
4. **`kernel` 和 `traced`**：未追踪与追踪的时间、时钟，以及被追踪的是哪些 SIMD。
5. **`statements`**：循环按 `kernel.tk` 行划分的每次行程周期数。`[steady]` 是循环的稳态；`[prologue]`、`[head]`、`[drain]` 和 `[epilogue]` 是其前后的部分。

之后还有其他各节：按类别划分的时间、每个 SE 的占用率、主循环按桶划分的每次行程周期数、停顿最多的指令、等待及约束它们的加载，以及最热的指令。

## 结论类别及其调节手段 {#verdict-classes-and-their-levers}

阈值是启发式的，作为常量保存在 `runtime/src/sqtt/verdict.rs` 中。

| 类别 | 触发条件 | Knob | tk 中的调节手段 |
|---|---|---|---|
| `matrix_bound` | 矩阵流水线繁忙 ≥ 稳态窗口的 70% | `less_matrix_work` | 更便宜的类型或更少的矩阵操作；问题不在调度 |
| `issue_bound` | 某条流水线繁忙 ≥ 90%，且矩阵流水线低于其 60% | `cut_issue` | 从循环中去掉非矩阵指令 |
| `occupancy_limited` | SIMD 空闲 ≥ 10% | `free_resources`、`more_workgroups`、`more_waves` | 每个工作组更少的 VGPR 或更少的 LDS；更小的 tile 或更多拆分以获得更多工作组 |
| `load_exposed` | 全局加载的等待和停顿 ≥ 循环的 10%，且为暴露的 | `latency`、`move_use` | 更深的环形缓冲，或提前一步发出的加载；`move_use` 指出紧跟在加载之后的使用 |
| `lds_exposed` | LDS 等待和停顿 ≥ 10%，且为暴露的 | `lds_flush`、`lds_use`、`lds_queue` | 更少、更宽的 `ds` 操作，更早读取下一个分片，无冲突的布局 |
| `barrier` | 屏障周期 ≥ 循环的 10% | `barrier` | 先解决其他发现；只有当屏障的暴露份额超过 10% 时才删除、合并或拆分它 |
| `loop_overhead` | VALU+SALU 发出 ≥ 矩阵发出的 10%，或存在任何发散分支 | `hoist` | 将地址和边界检查的运算提出循环；使边界检查一致（uniform） |

屏障是一个汇点：各 wave 停在那里等待最慢的 wave，因此它的时间属于让那个 wave 变慢的任何原因。该发现会列出最后到达的是谁、在哪条指令之后到达。

标记为 `approx` 的发现，其数字需要谨慎解读：`perturbed`、`untimed_matrix`、`approx_bindings`（猜测的绑定）、`implausible_time`（损坏的流）、`truncated` 或 `per_iteration`。

## `--attribute` 与 `--compare` {#--attribute-and---compare}

`--attribute` 仅适用于 GEMM，因为只有 `tk_bench sqtt gemm` 的采集带有代价模型的价格。它把每个计价项（`steps`、`reads`、`fill`、`wait`、`lone_trip`、`trip`）与其所计价指令的追踪周期并列。各项相互重叠，因此应比较份额和每次行程的节奏，而不是总数。

`--compare A B` 将 B 的采集与 A 的对比（Δ = B − A）：

```bash
cargo run --release -p svod-tk --example tk_bench -- sqtt --compare A B [--json] [--top N]
```

它打印两者的时间、出现、消失或变化的发现、按 |Δ exposed| 排序的每次行程循环桶、按文本对齐的语句，以及按助记符对齐的循环指令。当两份采集运行在不同的功耗状态、其追踪/未追踪比值相差超过 10%，或其行程覆盖的归约量不同时，它会以 `!!` 发出警告。没有主循环的采集没有可对齐的内容。

## 工作流程 {#the-workflow}

1. 在未追踪剖析的逐内核表中找到慢的内核。
2. 追踪它：`tk_bench sqtt <op> <shape>`；对于只存在于模型内部的内核，用带 `SqttOptions` 的 `Tensor::profile`。
3. 阅读 `ceiling` 和 `bound`。`matrix_bound` 的内核已接近其上限：改动工作量，而不是调度。
4. 取第一项发现、它的证据和它的 `@` 语句。该语句就是 tile 程序中要改动的那一行。
5. 做一处改动，以指向新目录的 `--out` 再次追踪，并运行 `--compare old new`。检查目标桶是否发生了变化，并只依据未追踪时间判断收益。
6. 保留改动之前，在空闲的 GPU 上用 `tk_bench` 或调优器确认。

## 实例 {#worked-examples}

### 注意力：提前一步加载偏置 {#attention-loading-the-bias-a-step-ahead}

一个直接读取的注意力偏置 tile 是在使用它的分数的那一步中加载的，因此它的等待紧跟在加载之后。这正是 `load_exposed` 以 `move_use` 指出的模式。现在内核提前一步加载每个 tile（`tk/src/kernels/attention.rs` 中由 `pipeline_with` 将其带入下一步），因此加载有整整一步的时间完成。PR #230 测得 d128 下 −19%、d64 下 −7%。

### Nearest：在 RDNA4 上拆分屏障 {#nearest-splitting-the-barrier-on-rdna4}

`rdna4_nearest_joined` 和 `rdna4_nearest_split` 夹具是同一个 `NearestCfg`（n4096 d128 k16），分别为采用 RDNA4 拆分屏障之前和之后。在 gfx12 上，寄存器暂存环形缓冲的稳态循环现在在 LDS 提交之后发出屏障信号（`arrive`，`s_barrier_signal`），并在行程末尾等待它（`await`，`s_barrier_wait`）。两者之间的尾部在较慢的 wave 追赶时运行。对这两个夹具运行 `--compare` 即可看到：

```text
time       180320 -> 144840 ns untraced (-19.7%); traced/untraced 1.01x -> 1.02x; …
changed    barrier            46.8% ->  18.6%
appears    load_exposed           - ->  13.0%
bucket                A raw    B raw    Δ raw    A exp    B exp    Δ exp
barrier              1081.7    255.1   -826.6    962.9    202.6   -760.2
   A raw    B raw    Δ raw    A exp    B exp    Δ exp  statement
  1081.7      0.0  -1081.7    962.9      0.0   -962.9  barrier [steady]
     0.0    255.1   +255.1      0.0    202.6   +202.6  await [steady]
```

屏障暴露从每次行程 963 个周期降到 203 个周期，`barrier [steady]` 语句变为 `arrive` + `await [steady]`，未追踪时间从 180 µs 降到 145 µs。每一侧都是单次追踪运行。PR #230 的基准测试测得 nearest d128 k16 为 −5.6%。新出现的 `load_exposed` 发现是下一个调节手段：屏障消失后，全局加载的延迟显露出来。

### GEMM：不规整的 M {#gemm-a-ragged-m}

`rdna4_gemm_1536` 和 `rdna4_gemm_1500` 夹具是同一个 GEMM 配置，M 分别为 1536 和 1500（N 1280、K 5120，Whisper 的 fc2）。两次重放都给出 `bound matrix_bound`。只有 1500 那次额外出现 `loop_overhead`，其证据是主循环中的八个发散分支：

```text
 2 loop_overhead       0.5% (raw 0.7%)  valu 24 + salu 7 issue cycles an iteration beside matrix 443 (7%), 22 valu ops an iteration, 8 divergent branches (approx: per_iteration)
   0x5f0 s_and_saveexec_b32 s1, vcc_lo
   0x5f4 s_cbranch_execz 3
   knob: hoist address and guard arithmetic out of the loop; make guards uniform (no exec branches)
```

M 不是 tile 的倍数时，边界检查会以 exec 分支的形式出现在循环内部；而 1536 的采集是其 128 行 tile 的倍数，没有这样的分支。这个结论是定性的：不规整的边缘会耗费循环指令，而填充到 tile 倍数的形状不需要付出这一代价，因此在为不规整形状调整调度之前，先检查填充后的形状。

## 注意事项 {#caveats}

- 追踪只能看到两个 SIMD。从不落到它们上的网格在那里什么也追踪不到。
- 追踪下的时间绝不是基准测试结果。用 `tk_bench` 或调优器在不追踪的情况下评判一处改动。
- `exposed` 是对其他流水线所掩盖部分的估计，因此并不保证一处修复能节省的时间。
- 阈值是启发式的。低于 10% 的发现是证据，而非结论。
- 没有语句时（图内核，或代码对象没有行号表的采集），数字按每次循环迭代而非每次行程计算。
- 来自旧 `meta.json` 格式的采集无法加载；请重新采集。
