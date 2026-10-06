---
sidebar_label: 剖析
---

# CUDA 上的剖析

[分层 profiler](../../tile-kernels/profiling.md) 在 `DispatchTimestamps` 与
`KernelResources` 句柄之上是后端中立的。本页讲的是 CUDA 后端往那些句柄里
放了什么，以及有哪些层级存在。

| 层级 | 在 CUDA 上 | 来源 |
|---|---|---|
| **1 — 设备时间** | 有 | 环绕每次启动的 CUDA event 对 |
| **2 — roofline** | 有 | 后端中立（IR FLOP 估算、plan 的缓冲区） |
| **3 — 静态占用率** | 有 | `cuFuncGetAttribute` + `cuOccupancyMaxActiveBlocksPerMultiprocessor` |
| **4 — 硬件计数器** | 有 | CUPTI range profiler（`libcupti.so.13`） |

```bash
SVOD_DEVICE=CUDA:0 cargo run --release -p svod-model --example gigaam_infer -- --profile ./audio.wav
```

---

## 第 1 层：event 时间戳

设置了 `profile` 的 `CudaPlanCtx::dispatch` 会在 plan 的流上、于启动之前与
之后各记录一个**计时 event**，并返回一个同时持有两者的
`CudaDispatchTimestamps`。`timestamps_ns` 必须报告 GPU 时钟上的纳秒，所以它
这样计算：

```text
start    = cuEventElapsedTime(base_event, start_event)   // ms since the device opened
duration = cuEventElapsedTime(start_event, end_event)
end      = start + duration
```

基准 event 在 `CudaDevice::open` 时记录一次，是这条 timeline 的零点。持续时间
是在这一对之间直接测得的（完整的 event 分辨率，约半微秒）；而绝对位置要过
一个 `f32` 毫秒计数，它随着进程变老而变粗，这也正是 `end` 由 `start` 推导
而来、而不是同样对着基准去测量的原因。两个 event 都必须已经完成
（`cuEventQuery`），否则句柄报告 `None`。

图重放以同样的方式被剖析：`replay_profiled` 运行一个链式可执行体，在每个
内核之前和之后各有一个 event-record 节点，并为每个被捕获的内核返回一个句柄
（[架构](./architecture.md)）。

BEAM 所用的 `Program::execute_timed` 是调度流上的同一对 event，以
`Duration` 的形式返回。

---

## 第 3 层：静态资源

`CudaProgram::resource_usage` 用加载时读到的函数属性填充
`KernelResources`：

| 列 | 字段 | 来源 |
|---|---|---|
| `VGPR` | `vgprs` | `CU_FUNC_ATTRIBUTE_NUM_REGS`（每线程寄存器数） |
| `SGPR` | `sgprs` | `-`（NVIDIA 上没有标量寄存器堆） |
| `LDS` | `lds_bytes` | `CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`（静态 `.shared`） |
| `scratch` | `scratch_bytes` | `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`（每线程的 `.local`） |
| `occ%` | `occupancy` | `cuOccupancyMaxActiveBlocksPerMultiprocessor(block) × block / 每 SM 最大线程数`，上限截断为 1 |

`wave_size` 就是设备的 warp 大小（32）。占用率查询需要一个 block 尺寸：程序
记住了它**最近一次启动**的 block，在任何启动之前则回落到函数的
`maxThreadsPerBlock`。与仅受寄存器限制的 AMD 数字不同，驱动给出的计数已经把
寄存器、共享内存和每 SM 的 block 上限都折算了进去。

---

## 第 4 层：硬件计数器

计数器来自 CUPTI 的 range profiler，运行期从唯一一个 soname —— `libcupti.so.13`
—— 绑定：先在加载器路径上查找，再到 `/opt/cuda/lib64`、
`/usr/local/cuda/extras/CUPTI/lib64` 和 `$CUDA_PATH/{lib64,extras/CUPTI/lib64}`
里找（`device/src/cuda/cupti.rs`）。CUDA 13 把 PerfWorks 的 host API 并入了 CUPTI，
因此整条调用序列由这一个库承载 —— CUPTI 自己会 `dlopen`
`libnvperf_host.so`，所以它必须能被动态链接器解析到，而解析不到时的样子就是一个
`CUPTI_ERROR_NOT_INITIALIZED`。这个绑定和 `ptxas` 一样是
可选的：库缺失、不可用，或用 `SVOD_CUDA_CUPTI=0` 显式关闭时，`pmc_available()`
为 `false`，profiler 退化到第 1-3 层并打印它那行提示。

其中两个 params 结构体在 CUDA 13.3 中变大了，所以每次调用都先送出最新的
`struct_size`，遇到 `CUPTI_ERROR_INVALID_PARAMETER` 就退回一档 ——
`cuptiProfilerGetCounterAvailability`（41 然后 40）与
`cuptiProfilerHostInitialize`（56 然后 48）—— 而 `abi_ladder` 会记住已安装的
CUPTI 第一个没有以参数错误拒绝的尺寸。

`SVOD_PMC=1` 选择该后端的默认集合：

| 令牌 | 指标 | 含义 |
|---|---|---|
| `cycles` | `sm__cycles_active.sum` | 至少有一个 warp 驻留的周期数 |
| `warps` | `sm__warps_launched.sum` | 启动的 warp 数 |
| `inst` | `smsp__inst_executed.sum` | 执行的 warp 指令数 |
| `tensor` | `sm__pipe_tensor_cycles_active.sum` | tensor 流水线活跃的周期数 |
| `dram` | `dram__bytes.sum` | 经过 DRAM 的字节数 |

用令牌指定子集 —— `SVOD_PMC=tensor,dram`。令牌在各后端之间唯一，所以在 CUDA 上
写 AMD 的令牌只会被丢弃，而不会去错误地编程另一个模块；`set_pmc` 会返回它保留了
多少个，执行器则在 stderr 上把这件事说出来：

```text
SVOD_PMC: 2 of 5 requested counters are not collected on this backend
```

一个令牌全都无法识别的列表会回落到默认集合，而一个什么都没能存活下来的选择，
意味着一次普通的、未布防的计时运行。`tensor` 与 `cycles`
之比就是 matmul 或 flash-attention 内核的 tensor core 利用率；`dram` 则把受带宽
限制的内核和受发射限制的内核区分开。

```bash
SVOD_DEVICE=CUDA:0 SVOD_PMC=1 cargo bench -p svod-tk --bench matmul -- --profile-time 5
```

### 权限

驱动默认只允许管理员采集计数器，而且这个限制并不在你以为的地方生效：
`cuptiRangeProfilerEnable` 和 `cuptiRangeProfilerSetConfig` 没有权限也会成功，
只有计数器可用性镜像和 `cuptiRangeProfilerStart` 会返回
`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`。因此 `pmc_available()` 探测的正是这个
可用性镜像。解除限制：

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  | sudo tee /etc/modprobe.d/nvidia-profiling.conf
# 按你的发行版重建 initramfs，然后重启
```

`scripts/cupti_probe.cu` 会拿一个 saxpy 内核独立跑完整条序列并报告它停在哪一步，
这是区分权限问题和 toolkit 问题最快的办法；在 `sudo` 下跑一遍就能不重启地确认
这个判断。

### 采集的代价

采集运行在 `CUPTI_AutoRange` 加 `CUPTI_KernelReplay` 模式下：CUPTI 为每次启动
开一个 range，并在内部重放内核以覆盖多趟配置（上面这五个计数器在我们跑过的每一块
芯片上都是一趟就调度完的；需要更多趟的集合会在 `Stop` 处被抓住，并退化为仅有计时）。
两个后果已经替你处理好了：

- 被捕获的 CUDA graph 会作为一次不透明的提交重放，那样什么计数器都拿不到，
  所以带计数器的运行走逐次 dispatch 的路径。
- 内核重放会把该次 dispatch 自己的 event pair 放大好几个数量级，所以带计数器的
  运行会额外做一趟不带计数器的 pass；`merge_min` 保留它的计时，同时保留带计数器
  那一趟的计数器。因此同一张表里的计时和计数器来自不同的 pass。

回读由主机驱动，且一个 session 不能与下一个重叠，所以带计数器的 dispatch 会就地
同步。任何 CUPTI 失败都只会让该次 dispatch 退化为仅有计时，而不会让整个运行失败。

若要做内核内的计时实验，`svod_codegen::llvm::nvptx::globaltimer()` 会构建一个
读取 `%globaltimer` 的 `CUSTOM` 节点，那是纳秒级的 GPU 时钟。
