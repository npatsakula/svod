---
sidebar_label: 内核搜索
---

# 内核搜索：启发式、BEAM 与 Tensor Core

经过 `apply_pre_optimization` 之后，内核是一个由 `Weak` 和 `Reduce` range 构成的循环嵌套。优化器通过对 `Scheduler` 应用 `Opt` 来决定这些循环如何执行 —— 哪些成为网格维度、工作组、warp、向量 lane、展开的循环体或 tensor core fragment。有两种策略来挑选 opt：手写启发式（默认）和 BEAM 搜索。源码：`schedule/src/optimizer/{scheduler,opts,heuristics,beam,tc,renderer,config}.rs`、`ir/src/opt.rs`。

## 调度器与动作空间

`Scheduler::new(ast, renderer)` 为内核中的 `RANGE`（extent > 1）建立索引，按 `(axis_type.priority(), axis_id)` 排序；当 renderer 具备 `has_local`（GPU）时，`convert_loop_to_global` 把出现在每个 `STORE` 中的 `Weak` 轴变成 `Global`，在 CPU 上则什么也不做。随后 `apply_opt(scheduler, opt, append)` 每次调用重写一个 range：

| `OptOps` | 作用 | 条件（`opts.rs`） |
|----------|--------|--------------------|
| `UPCAST(axis, n)` | 从 `Global`/`Local`/`Weak` 轴上拆出 `n` 个 lane 作为 `Upcast` | `n <= renderer.upcast_max`；`n = 0` 表示取整个轴 |
| `UNROLL(axis, n)` | 从 `Reduce`/`GroupReduce` 轴上拆出 `n` 次迭代作为 `Unroll` | `axis` 是 `unrollable_dims()` 的索引；`n <= 32` |
| `LOCAL(axis, n)` | 从 `Global`/`Weak` 轴上拆出一个工作组维度 | `has_local`，且之前没有 `NOLOCALS` |
| `GROUP(axis, n)` / `GROUPTOP(axis, n)` | 把 `Reduce` 轴按内侧 / 外侧拆分为 `GroupReduce`（通过共享内存的两阶段归约） | `has_local && has_shared`，不超过 `shared_max`，不嵌套在另一个 reduce 中，**一旦应用了 TC opt 就会被拒绝** |
| `THREAD(axis, n)` | 在可全局化的 `Global`/`Weak` 轴上建立 CPU 核心维度 | `has_threads`，尚无 `Thread` 轴，`n <= global_max[0]` |
| `SWAP(a, b)` | 交换两个 `Global` 轴 | 两者都是 `Global` —— 因此在 CPU 上从不适用，那里的轴保持为 `Weak` |
| `PADTO(axis, n)` | 把轴填充到 `n` 的倍数，并屏蔽尾部 | extent 为常量，不是 `Upcast`/`Unroll`/`Thread`，填充量低于工作量的 4 倍，单索引 `INDEX` |
| `NOLOCALS` | 设置 `dont_use_locals`，阻止之后的 `LOCAL`；gpudims 只以全局维度启动 | 尚无 `Local`/`Warp`/`GroupReduce` 轴 |
| `TC(axis_choice, tc_select, tc_opt, use_tc)` | 把矩阵乘映射到 tensor core | 必须是第一个 opt；见下文 |

`get_optimized_ast_with_naming` 展平 range 列表并附上 `KernelInfo { name, applied_opts, dont_use_locals }`；名称是 `r_`/`E_` 加上按 range 顺序排列的各 extent（[完整示例](../codegen/worked-example.md)中的 `r_8_16_4`）。

## 启发式（`hand_coded_optimizations`）

`hand_coded_optimizations(&mut scheduler, &HeuristicsConfig)` 按以下顺序应用（`heuristics.rs`）：

1. **`try_tensor_cores`** —— 若 `tc_enabled != Disabled`、renderer 具备 core，且（在 `TcOpt::Strict` 下）恰好只有一个 reduce 轴：先 `tc::detect_matmul`，然后在各轴选择上执行 `apply_with_axis_choice`，再执行 `apply_tc_tiling` —— `FixedStep`：用 5/4/3/2 中第一个能整除的数对 M 和 N 做 `UPCAST`，对 N 做 4 或 2 的 `LOCAL`；`LaneBudget { accum_max: 128 }`（CUDA sm75/80/89）：先 `tc_warp_tile_growth`，再做 `wave_size / tc.threads` 的 `LOCAL`。成功则返回。
2. **`apply_image_upcasts`** —— 图像缓冲区。
3. **`apply_matvec_fast_path`** —— `SVOD_MV*` 矩阵向量乘配置（`PADTO`、对小轴 `UPCAST`、尽力而为的 `GROUP`、`LOCAL`、`UPCAST`、`UNROLL`）。成功则返回。
4. **`try_grouped_reduction`** —— 对至多 2048 个元素的输出（无 local 时为 240）使用 `GROUPTOP(axis, 16)`；否则使用 **`try_warp_row_reduction`**（按 wave 大小 `GROUP` 加上 `UNROLL 4`）。若此时已存在 `GroupReduce` 轴，函数返回。
5. **`apply_masked_upcasts`** —— 大小为 2–7、乘积 ≤ 49 的被屏蔽轴。
6. **`apply_heuristic_upcasts`** —— 当输出有 ≥ 1024 个元素且 upcast 乘积低于 32 时，按 3 或 4 做 `UPCAST`，轴按 `(num_strides, sum_strides, axis, vector rank)` 排序。
7. **`apply_unroll`** —— reduce 轴 ≤ 32 时完全展开（当两个都 ≤ 3 时第二个也展开），否则 `UNROLL 4`。
8. **`apply_default_upcast`** —— 若尚未做过任何 upcast 或 unroll，则对最后一个可 upcast 的轴做 `UPCAST 4`。
9. **`apply_local_dims`** —— 轴 0 的 `LOCAL` 大小取 `[32, 16, 8, 4, 3, 2]`，其他轴取 `[16, 8, 4, 3, 2]`，累计预算 128，至多三个，并有 `PADTO` 回退。
10. **`apply_threading`** —— 仅 CPU：在 `Weak` 轴上按 `[32, 16, 12, 8, 6, 5, 4, 3, 2]` 做 `THREAD`，保持每线程至少 131072 个元素，并有 `PADTO` + `THREAD` 回退。

`HeuristicsConfig::from_env` 读取 `SVOD_TC`（0 禁用，2 仅形状，其他值启用）、`SVOD_TC_OPT`/`TC_OPT`、`SVOD_TC_SELECT`/`TC_SELECT`、`SVOD_MV*`、`SVOD_NOLOCALS`、`SVOD_THREADS`。分组归约的阈值是 `heuristics.rs` 中的常量；`SVOD_K_VECTORIZE` 和 `SVOD_NO_OUTPUT_UPCAST` 设置的字段在这条路径上无人读取。

## BEAM 搜索

`BEAM=N`（N > 0）选择 `OptStrategy::Beam { width: N }`。随后 `realize` 让内核经过 `beam_search_cached_remote(scheduler, config, compiler_identity, behavior_fingerprint, compile_wave, benchmark)`（`beam.rs`）；普通的 `optimize_kernel_with_config` API 没有编译并计时的闭包，因此回退到启发式。

搜索过程（`beam_search_remote_staged`）：

1. 从 `[(scheduler, Duration::MAX)]` 开始，并把启发式的结果作为额外的第一轮候选加入。
2. **扩展**：对每个 beam 成员，`generate_actions` 尝试 193 个 `BEAM_ACTIONS` 中的每一个（开启 `BEAM_PADTO` 时为 200 个）：`passes_prefilter`（轴存在；若存在 `0` 变体，则跳过数量等于轴大小的动作）、`apply_opt`、`validate_limits`（`upcast_prod / tc_up <= max_upcast`、`local_prod <= max_local`）。当 `enable_nolocals` 时，为每个成员追加 `NOLOCALS`。
3. **编译**：在工作进程池中编译候选；若某候选线性化后的算子数达到 `max_uops`，或编译超过 `compile_timeout_secs`，则在此被丢弃。
4. **过滤**：丢弃 `compute_ops` 超过本轮最少值 1000 倍的候选，然后按二进制（或源码）键去重。
5. **计时**：每个运行 `num_runs` 次，得分取最小值；单次运行超过当前最优的 3 倍就提前终止；全局大小上限为 65536，时间再按比例换算回来。
6. **保留**最好的 `beam_width` 个。当最佳时间的提升不再达到 `min_progress_ns`（或已经低于它）时停止；若有提升，beam 会坍缩为唯一的胜者进入下一轮。

动作列表（`BEAM_ACTIONS`）：`UPCAST` 数量 `[0,2,3,4,5,7]` × 轴 0..8（48），`UNROLL` `[0,4,7]` × 0..5（15），`LOCAL` `[2,3,4,8,13,16,29]` × 0..6（42）外加 `(0,32)` 和 `(6,2)`，`GROUPTOP` `[13,16,28,29,32,49,64,256]` × 0..3（24），`GROUP` `[0,4,8,16]` × 0..3（12），`TC`（一个 `tc_opt = 0` 动作加上 `TC_OPT` 下的九种轴选择），0..5 内的 `SWAP` 对（10），`THREAD` `[2,3,4,5,8,12,16,24,32,64]` × 0..3（30）。`BEAM_PADTO` 为轴 0..7 添加 `PADTO(axis, 32)`。

### 缓存

结果持久化在 `$SVOD_BEAM_CACHE_DIR/beam_cache` 处的 `sled` 数据库中，否则在 `~/.cache/svod/beam_cache`（`dirs::cache_dir()`）。键（`CacheKey`，schema 11）由结构化 AST 哈希加上 beam 宽度、设备、`renderer.cache_fingerprint()`、编译器标识、各项限制（`max_upcast`、`max_local`、`max_uops`、`num_runs`、`min_progress_ns`、`enable_nolocals`、`compile_timeout_secs`）、行为指纹（`transcendental`、`disable_fast_idiv`）以及动作空间的哈希构成。值是 `applied_opts` 列表；命中时用 `replay_opts` 重放，验证并测速一次，若失败则使其失效。`IGNORE_BEAM_CACHE=1` 绕过缓存，`clear_cache` 清空缓存。

### 环境变量

| 变量 | 默认值 | 含义 |
|----------|---------|---------|
| `BEAM` | 0 | beam 宽度；0 = 启发式 |
| `BEAM_UPCAST_MAX`、`BEAM_LOCAL_MAX`、`BEAM_UOPS_MAX` | 256、1024、3000 | `validate_limits` 以及工作进程的算子数上限 |
| `BEAM_RUNS` | 3 | 每个候选的计时运行次数 |
| `BEAM_MIN_PROGRESS` | 10（µs，以 ns 存储） | 停止阈值 |
| `BEAM_PADTO` | 0 | 添加七个 `PADTO` 动作 |
| `NOLOCALS` / `SVOD_NOLOCALS` | 未设置 | 添加 `NOLOCALS` 动作 |
| `PARALLEL` | 0 | 编译工作进程数（GPU 默认为线程预算，其他为 1） |
| `BEAM_TIMEOUT_SEC`、`BEAM_MAX_TASKS_PER_CHILD` | 10、16 | 工作进程看门狗与回收 |
| `TC`、`TC_OPT` | 1、2 | BEAM 的 tensor core 动作（BEAM 下忽略 `TC_SELECT`：总是 `Auto`） |
| `BEAM_DEBUG`、`BEAM_LOG_SURPASS_MAX` | 未设置 | 诊断 |
| `IGNORE_BEAM_CACHE`、`SVOD_BEAM_CACHE_DIR` | 未设置 | 缓存控制 |

:::tip[BEAM 不读取启发式开关]
BEAM 内部的启发式种子使用 `HeuristicsConfig::from_env()`，但搜索自身的 TC 动作读取的是 `TC` 和 `TC_OPT`，而不是 `SVOD_TC`/`SVOD_TC_OPT`。`SVOD_NOOPT`（任意值）选择 `OptStrategy::None`：完全不应用 opt，但预优化和 post-optimization 仍会运行。
:::

## Tensor core

`renderer.rs` 按 `RendererDevice` 保存 core 表；维度为 `(N, M, K)`：

| 目标 | Core（输入 → 输出） | 线程数 |
|--------|------------------|---------|
| CUDA sm75 | 8×16×8 f16→f32、f16→f16 | 32 |
| CUDA sm80 | 8×16×16 f16→f32、bf16→f32、f16→f16；8×16×8 f16→f32、f16→f16；8×16×32 i8→i32；可选 tf32 8×16×8 | 32 |
| CUDA sm89 | sm80 外加 8×16×32 fp8 e4m3/e5m2→f32 | 32 |
| AMD RDNA3 | 16×16×16 f16→f32、f16→f16、bf16→f32、i8→i32 | 32 |
| AMD RDNA4 | RDNA3 外加 bf16→bf16 | 32 |
| AMD CDNA3 | 16×16×32 fp8 e5m2/e4m3；16×16×16 f16/bf16→f32 | 64 |
| AMD CDNA4 | CDNA3 外加 16×16×128 fp8 | 64 |
| Metal | 8×8×8 f32/f16/bf16 各变体 | 32 |
| Intel Xe | 8×8×16 f16→f32 | 8 |
| WebGPU、CPU | 无 | — |

当该计算能力具备 bf16 mma 时，`for_cuda_arch` 选择 sm80 配置，否则选择 sm75，sm75 以下则没有 core。

`tc.rs`：`detect_matmul` 寻找 `REDUCE(Add, MUL(in0, in1), reduce_ranges)`；只有 `in0` 使用的 range 是 M 的候选，只有 `in1` 使用的是 N，reduce range 是 K，每个 `(M, N, K)` 三元组就是一种轴选择（本身是 `Reduce` 轴的 M/N range 会被拒绝）；`select_tensor_core` 匹配输入和输出的标量 dtype（没有原生 core 的 fp8 输入回退到 f16 core）；`apply_with_axis_choice` 在 64 次尝试的预算内遍历轴选择 × core。应用一个 core 的方式是拆分轴：一个 extent 为 `tc.threads` 的 `Warp` range，每个 `TcOpt::Upcast` 条目对应一个大小为 2 的 `Upcast` 轴，每个 `TcOpt::Local` 条目取 warp 索引的一位（`warp % 2`、`warp / 2`），K 变成 `log2(K)` 个大小为 2 的 `Unroll` 轴；剩余的 N/M 保持为 `Global`，剩余的 reduce 轴把 `WMMA` 包在一个 `REDUCE` 中。`TcUsage::ShapeOnly`（`SVOD_TC=2`）执行拆分，但不生成 `WMMA`。

`TcOpt` 级别（`TC_OPT`）：**0 Strict** —— 只允许一个 reduce 轴，M/N/K 必须能整除；**1 Relaxed** —— `tc.rs` 内部采用相同的整除规则；**2 Padded**（默认）—— 当填充增加不超过 25% 时，对不能整除的轴做 `PADTO`；**3 Unbounded** —— 在 `PADTO` 自身 4 倍限制之下填充。符号轴从不使用 tensor core。

## 以编程方式配置

```rust
use svod_schedule::optimizer::{OptStrategy, OptimizerConfig};
use svod_tensor::PrepareConfig;

let config = PrepareConfig::from(
    OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 8 })
        .build(),
);
tensor.realize_with(&config)?;
```

`OptimizerConfig`（`bon` builder）包含 `strategy`、`beam: BeamConfig`、`heuristics: HeuristicsConfig`、`transcendental`（`TRANSCENDENTAL`，默认 1；≥ 2 强制使用多项式分解）、`disable_fast_idiv`（`DISABLE_FAST_IDIV`，默认 **1**：魔数除法需通过 `DISABLE_FAST_IDIV=0` 主动开启）以及 `opts_to_apply`（显式的 opt 列表，也可从内核 `SINK` 的 `KernelInfo` 中读取；它会覆盖策略，且应用失败的 opt 会报错）。`PrepareConfig` 包含 `optimizer`、`planner_mode`、`disable_schedule_cache`、`device_local_outputs`、`threads`，以及构造函数 `Default`、`from_env`、`device_local`、`for_cpu_backend`、`for_{amd,metal,cuda}_if_available`。
