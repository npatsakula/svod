---
sidebar_label: 架构
---

# 架构

本页跟随后端从驱动绑定一路走到图重放。除非另有说明，下面的一切
都在 `device/src/cuda/` 中。

```text
mod.rs        the dl_api! runtime-binding macro and the module's re-exports
sys.rs        the bound driver entry points (libloading), CUresult, handles
device.rs     CudaDevice: primary context, limits, lanes, base event, scoped-sync tables, poison latch
allocator.rs  CudaAllocator: device / managed / pinned memory, staged copies
program.rs    CudaProgram: cubin or PTX module load, cuLaunchKernel, execute_timed, resources
sync.rs       CudaPlanCtx, CudaDispatchTimestamps, CudaCompletionToken
graph.rs      CudaGraph: a CUDA graph DAG from GraphKernel::deps, patched replays
cupti.rs      the CUPTI range profiler behind Tier 4 (see Profiling)
```

---

## 驱动绑定

`sys.rs` 在一个 `dl_api!` 块里声明了后端用到的每一个入口点：Rust 字段、
确切的导出名，以及 C 原型；宏本身住在 `mod.rs` 里，`cupti.rs` 也复用它。
`Api::load` 打开 `libcuda.so.1` 并预先把它们全部解析出来，因此一个缺失或
被改名的符号会在加载时一次性失败，表现为
`Error::DeviceUnavailable`（`libcuda.so.1 has no symbol ...`），而不是在首次
使用时才失败。这些名字是 `cuda.h` 会重映射到的**带版本的**导出：
`cuMemAlloc_v2`、`cuDevicePrimaryCtxRelease_v2`、`cuGraphAddKernelNode_v2`、
`cuGraphExecKernelNodeSetParams_v2`、`cuGraphInstantiateWithFlags`（不带版本的
`cuGraphInstantiate` 是一个遗留的五参数 ABI，绝不会被碰到）。
`cuEventElapsedTime` 是刻意的例外：它的 `_v2` 是 CUDA 12.8，那会把驱动下限
抬到 R570。

句柄是 `#[repr(transparent)]` 的指针 newtype（`CUcontext`、`CUmodule`、
`CUfunction`、`CUstream`、`CUevent`、`CUgraph`、`CUgraphExec`……）；
`CUdeviceptr` 是 `u64`。`CUresult` 是一个整数 newtype，因此来自更新驱动的
错误码仍能原样往返；`CUresult::check("cuLaunchKernel")` 会把一次失败变成

```text
CUDA cuLaunchKernel failed: CUDA_ERROR_INVALID_VALUE (1): invalid argument
```

用的是驱动自己的 `cuGetErrorName` / `cuGetErrorString`。
`CudaKernelNodeParams` 结构体镜像 `CUDA_KERNEL_NODE_PARAMS_v2`，并带有
编译期的大小与偏移断言。

---

## 设备、上下文、流

`CudaDevice::open(id)` 每进程缓存一次。它运行 `cuInit`，保持住设备的
**主上下文**（`cuDevicePrimaryCtxRetain`），读取它需要的 `CudaLimits`
（`cuDeviceGetAttribute`：SM 数量、每 block 与每 SM 的线程数、每 SM 的 block 数、
每 block 的共享内存、warp 大小，以及托管内存是否既受支持又可被并发访问），创建两个非阻塞流
（供分配器用的**复制流**，以及供每调用 `Program::execute` 用的**调度流**），
并在旧式（legacy）默认流上记录一个**基准 event**——在返回设备之前会等待它——
它是每个 GPU 时钟时间戳的零点。

驱动按线程保存当前上下文，因此后端的每一个入口点都以 `enter()` 开始：
若设备已被毒化则拒绝，然后 `cuCtxSetCurrent`。一个**粘性**的 `CUresult`
（`ILLEGAL_ADDRESS`、`LAUNCH_FAILED`、`ILLEGAL_INSTRUCTION`、`MISALIGNED_ADDRESS`、
`ECC_UNCORRECTABLE`、`LAUNCH_TIMEOUT`、`ASSERT`、`HARDWARE_STACK_ERROR`、
`INVALID_ADDRESS_SPACE`、`INVALID_PC`、`DEINITIALIZED`，以及出于保守考虑的
`UNKNOWN`——见 `sys.rs`）会连同它的
消息闩上 poison 标志；此后该设备上的每一次调用都会带着那条消息快速失败，
与 AMD 上一样。

---

## 内存

一个 `RawBuffer::Cuda` 携带一个设备指针、一个可选的宿主指针，以及它的
`CudaMemory` 种类，后者依据 `BufferSpec` 选出：

| `BufferSpec` | 种类 | 驱动调用 |
|---|---|---|
| 默认 | `Device` | `cuMemAlloc`——设备内存，没有宿主映射 |
| `cpu_access` | 若设备报告支持并发的托管访问则为 `Managed`，否则为 `Pinned` | `cuMemAllocManaged(ATTACH_GLOBAL)`，一个地址在两侧都有效；否则为 `cuMemHostAlloc(PORTABLE \| DEVICEMAP)` + `cuMemHostGetDevicePointer` |
| `host` | `Pinned` | `cuMemHostAlloc(PORTABLE \| DEVICEMAP)`，内核经由总线读取它 |

`supports_device_local()` 为 `true`，因此中间结果留在设备上。
宿主 <-> 设备的复制先等待该存储在飞行中的生产者与读者
（`CudaDevice::wait_storage`，见下——宿主访问并不与那些车道相互定序），
然后再搬运数据。不超过 4 MiB 时，一次 copy-out 是一个同步的
`cuMemcpyDtoH`，而一次 copy-in 是复制车道上的一个 `cuMemcpyHtoDAsync`，
并被发布为该存储新的生产者；只有当源是驱动会跟踪的内存（固定、已注册或
托管，用 `cuPointerGetAttribute` 问出来）时它才同步该流，因为驱动在返回
之前会先把可分页的源暂存一遍。超过 4 MiB 时两个方向都以 4 MiB 为块、
通过一个惰性分配的**固定（pinned）暂存缓冲区**用 `cuMemcpyHtoDAsync` /
`cuMemcpyDtoHAsync` 搬运，每块同步一次该流。固定缓冲区则直接 `memcpy`。
设备到设备的 `_transfer` 与清零在复制车道上是异步的：用 `cuStreamWaitEvent`
排在那些生产者之后，被发布为两个范围新的生产者，并被此后任意车道上的每一次
启动等待，因此它们不会阻塞宿主——例外是一次分配内部相互重叠的范围：它会经由一个临时
缓冲区中转以保持 `memmove` 语义，并在释放该临时缓冲区时等待。释放会先等待该存储的生产者；若等待失败
（上下文已被毒化），该分配会被**隔离**（泄漏），而不是在一个仍在飞行中的
内核之下被释放。与每个计算分配器一样，它坐落在 `LruAllocator` 之下，而后者
会把一个被回收的分配栅栏在其上一位所有者的生产者上。

---

## 程序与启动

`CudaProgram::load` 按 `is_cubin` 分支——一个 ELF 映像走 `validate_cubin`，
PTX 文本走入口的 `.param` 检查——两者都抵达同一个 `cuModuleLoadDataEx`，
带着 16 KiB 的错误与信息日志缓冲区，因此一次 JIT 失败会浮现为
`Error::CudaJit { kernel, cause, log }`，其中携带 `ptxas` 自己的消息（见
[调试](./debugging.md)）；信息日志则走 `tracing::debug!`。随后它用
`cuModuleGetFunction` 绑定入口，并读取函数属性
`MAX_THREADS_PER_BLOCK`、`NUM_REGS`、`SHARED_SIZE_BYTES`、`LOCAL_SIZE_BYTES`。
模块与任何捕获了它的图以 `Arc` 共享，并在最后一次 drop 时卸载。

内核参数作为**一整块打包的 blob** 经由 `cuLaunchKernel` 的 `extra` 数组
（`CU_LAUNCH_PARAM_BUFFER_POINTER` / `_SIZE` / `_END`）传递，由共享的
`ClikeKernargLayout` 布置：8 字节的设备指针、4 字节的 `i32` 标量，按 PARAM
槽顺序排列，这恰好就是 PTX 天然的 `.param` 布局。`global_size` 是**以 block
为单位的 grid**，`local_size` 是**以线程为单位的 block**（AMD 与 Metal 所用的
工作组约定）；一个大于函数 `maxThreadsPerBlock` 的 block 会在启动前被拒绝，
消息中带上寄存器、共享内存与局部内存的数字。

`Program::execute` 在设备的调度流上启动，并可选地在其上等待；
`execute_timed` 在启动前后记录一对计时 event 并返回 `cuEventElapsedTime`，
因此 BEAM 按 GPU 时间对候选排名。

---

## plan 上下文、令牌、timeline

每个执行 plan 得到一个 `CudaPlanCtx`：**一个非阻塞流**，那就是它的车道，
外加装填计数器时的 CUPTI 计数器选择与 session。`dispatch` 在其上启动；
设置了 `profile` 时，它会用计时 event 把这次启动括起来，并返回一个
`CudaDispatchTimestamps`（[剖析](./profiling.md)）。
`completion_token` 记录一个仅完成用的 event（`CU_EVENT_DISABLE_TIMING`），
它的 `wait` 是 `cuEventSynchronize`，`retired` 是 `cuEventQuery`；
`synchronize` 是 `cuStreamSynchronize`。

### 带作用域的同步

各条车道之间并不相互定序，因此 `CudaDevice` 维护三张表
（`device/src/cuda/device.rs` 的模块文档）：

- **producers**——存储基址 -> 每条车道上读过或写过它的最新完成令牌，以及它是否写过
  （一次宿主覆写对在飞行中的读者也是一个 WAR 冒险）。执行 plan 在每次 execute
  之后，把它自己或其图的令牌发布到它触及的每一处存储上，并在它写的存储上标记为写；分配器在
  每次传输或 memset 之后发布一个复制车道的令牌。`wait_storage(base)` 先排空
  下面说的那些车道，再等待这些令牌，然后把它们从表里丢掉。一处表里不认识的
  存储——包括其最新令牌属于另一个后端的那种——会回落到 `cuCtxSynchronize`。
- **lanes**——每一条活着的车道，以及它上面有多少次尚无令牌被发布的提交
  （每调用的 `Program::execute`、一个中途失败的 plan、一次在其令牌被取走
  之前的图重放）。`wait_storage` 会在宿主上排空这样的车道；而一次复制则改为
  在每条车道上记录一个尾部 event 并在 GPU 上等待它。复制车道自己不在这张表里。
- **copy tail**——最新的那个复制车道 event；每一次启动都会在 GPU 上先等待它
  再运行，因此异步复制先于此后的每一个内核。

每次 execute 之前，plan 用 `cuStreamWaitEvent` 把自己的车道（或其图的车道）
排在其他车道与之冲突的令牌之后（`CudaDevice::order_lane_after`）：它读的每处存储的
写者，以及它写的每处存储的全部令牌。只读同一存储的 plan（例如共享权重）仍然并发。
令牌比车道活得长，所以一次 plan 已被丢弃的一次性 `realize` 仍会给下一次定序。

`SVOD_CUDA_SCOPED_SYNC=0` 会把这一切统统关掉：每一次等待都排空上下文，
每一次复制都同步复制流，每一次 plan execute 都先排空上下文。

执行器的跨 plan 定序在每个后端上都是一个宿主信号（`CpuTimelineSignal`），
CUDA 也不例外；并没有它自己的 `TimelineSignal` 实现。把这个宿主信号挡在
关键路径之外的，正是上面这套机制：那些表用 `cuStreamWaitEvent` 让 GPU 工作
相对 GPU 工作定序，因此一个 plan 在宿主上等待的，永远只是它真正要读的东西。

---

## 图

`CudaGraph::capture` 把一条被捕获的内核链变成一张真正的 **CUDA 图**：
每个内核一个 `cuGraphAddKernelNode_v2`，其依赖列表恰好就是
`GraphKernel::deps`，即宿主侧的冒险分析。因此相互独立的内核可以在设备上
重叠执行（AMD 后端丢弃 `deps`，因为单条顺序环让它们变得多余）。每个节点的
params 经由与即时启动相同的 `extra` 协议指向该内核的 kernarg blob；图用
`cuGraphInstantiateWithFlags` 实例化。对于空链、非 CUDA 程序，或另一个设备的
程序，捕获会谢绝（`Ok(None)`）；在设备已被毒化或依赖于一个更靠后的内核时，
捕获会失败（`Err`）；plan 对这两种情况都按每调用调度处理。

`replay(buffers, vals)` 只重新打包那些 `(buffers, vals)` 切片发生了变化的
内核，并用 `cuGraphExecKernelNodeSetParams_v2` 更新那些节点，然后在图自己的
流上 `cuGraphLaunch`。有一个微妙之处：记录下来的冒险只对捕获时的那套
**别名关系**有效。如果一次重放绑定的缓冲区使得另外一对槽位现在共享了同一个
地址，图就会切换到一条惰性构建的**捕获顺序链**（每个内核排在前一个之后），
那总是正确的。

`replay_profiled` 使用第三个可执行体，即在每个内核前后各带一对
`cuGraphAddEventRecordNode` 的那条链；这些 event 每次启动都会重新装填
（`cuGraphExecEventRecordNodeSetEvent`），因此已经发出去的句柄仍保有它们的
时间戳，而每个被捕获的内核会按捕获顺序返回一个 `CudaDispatchTimestamps`。

---

## 对象缓存标识

编译出的 PTX 会走共享的磁盘对象缓存，以渲染出的 IR 和一个
`CompilerIdentity` 为键：

```text
schema:              OBJECT_CACHE_SCHEMA
backend:             nvptx-clang
target_architecture: nvptx64-nvidia-cuda/sm_86
toolchain:           <clang identity>[;ptxas:path=...;version=...]
flags:               -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 --cuda-feature=+ptx78 -Wno-override-module - -o -
                     [-arch=sm_86 -o /dev/stdout /dev/stdin]
abi:                 ptx-kernel-abi-v1;warp-size=32
object_format:       ptx-text-v1 | cubin-v1
```

方括号里的那两半是 `ptxas` 那条路径：有这个汇编器时，缓存下来的对象是一个
**cubin**；没有它时则是 **PTX 文本**，由驱动在加载时汇编，并把 SASS 留在
自己的 `~/.nv/ComputeCache` 里。两种格式绝不会共用同一条缓存项，因为它们在
`toolchain`、`flags` 与 `object_format` 上都各不相同。渲染出的 IR 也不是键的
全部：ABI 描述符会被追加到它后面，因为一个 cubin 的入口是在编译期对着它们
校验的。每一次缓存命中在抵达驱动之前，都会被其格式对应的校验器重新校验——
`validate_cubin` 或 `validate_ptx`，见[代码生成](./codegen.md)。
`SVOD_OBJECT_CACHE=0` 关闭该缓存，`SVOD_OBJECT_CACHE_DIR` 可迁移它的位置，
`SVOD_OBJECT_CACHE_MAX_BYTES` 则设定它的容量预算（见 [CPU 页面](../cpu.md)）。

设备工厂（`create_cuda_device`）还会拒绝一台每 block 共享内存上限低于优化器
profile 静态 `shared_max` 的设备，否则一个按该 profile 定尺的内核就只会在
JIT 时才失败。
