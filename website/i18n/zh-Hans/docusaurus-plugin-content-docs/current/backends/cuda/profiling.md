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
| **4 — 硬件计数器** | **无** | 需要 CUPTI；未做绑定 |

```bash
SVOD_DEVICE=CUDA:0 SVOD_PROFILE_ITERS=20 cargo run --release -p svod-model --example gigaam_infer -- ./audio.wav
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
| `occ%` | `occupancy` | `cuOccupancyMaxActiveBlocksPerMultiprocessor(block) × block / 每 SM 最大线程数` |

`wave_size` 就是设备的 warp 大小（32）。占用率查询需要一个 block 尺寸：程序
记住了它**最近一次启动**的 block，在任何启动之前则回落到函数的
`maxThreadsPerBlock`。与仅受寄存器限制的 AMD 数字不同，驱动给出的计数已经把
寄存器、共享内存和每 SM 的 block 上限都折算了进去。

---

## 第 4 层：不可用

这里没有 CUPTI 绑定，因此 CUDA 上的 `PlanContext::pmc_available()` 为
`false`。设置 `SVOD_PMC=1` 不会失败：profiler 会退化到第 1-3 层，并打印它
那行说明计数器不可用的提示。`PmcCounter` 枚举今天是 AMD-SQ 专属的；把它拓宽
是[路线图](./limitations.md)的一部分。

若要做内核内的计时实验，`svod_codegen::llvm::nvptx::globaltimer()` 会构建一个
读取 `%globaltimer` 的 `CUSTOM` 节点，那是纳秒级的 GPU 时钟。
