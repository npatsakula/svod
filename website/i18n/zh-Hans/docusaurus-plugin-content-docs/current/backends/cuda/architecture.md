---
sidebar_label: 架构
---

# 架构

本页跟随后端从驱动绑定一路走到图重放。除非另有说明，下面的一切
都在 `device/src/cuda/` 中。

```text
sys.rs        the bound driver entry points (libloading), CUresult, handles
device.rs     CudaDevice: primary context, limits, streams, base event, poison latch
allocator.rs  CudaAllocator: device / managed / pinned memory, staged copies
program.rs    CudaProgram: PTX module load, cuLaunchKernel, execute_timed, resources
sync.rs       CudaPlanCtx, CudaDispatchTimestamps, CudaCompletionToken, CudaTimelineSignal
graph.rs      CudaGraph: a CUDA graph DAG from GraphKernel::deps, patched replays
```

---

## 驱动绑定

`sys.rs` 在一个 `cuda_api!` 宏里声明了后端用到的每一个入口点：Rust 字段、
确切的导出名，以及 C 原型。`Api::load` 打开 `libcuda.so.1` 并预先把它们
全部解析出来，因此一个缺失或被改名的符号会在加载时一次性失败，表现为
`Error::DeviceUnavailable`（`libcuda.so.1 has no symbol ...`），而不是在首次
使用时才失败。这些名字是 `cuda.h` 会重映射到的**带版本的**导出：
`cuMemAlloc_v2`、`cuCtxDestroy_v2`、`cuGraphAddKernelNode_v2`、
`cuGraphExecKernelNodeSetParams_v2`、`cuGraphInstantiateWithFlags`（不带版本的
`cuGraphInstantiate` 是一个遗留的五参数 ABI，绝不会被碰到）。

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
（`cuDeviceGetAttribute`：SM 数量、每 block 与每 SM 的线程数与共享内存、
寄存器、warp 大小、L2、托管内存支持），创建两个非阻塞流（供分配器用的
**复制流**，以及供每调用 `Program::execute` 用的**调度流**），并记录一个
**基准 event**，它是每个 GPU 时钟时间戳的零点。

驱动按线程保存当前上下文，因此后端的每一个入口点都以 `enter()` 开始：
若设备已被毒化则拒绝，然后 `cuCtxSetCurrent`。一个**粘性**的 `CUresult`
（`ILLEGAL_ADDRESS`、`LAUNCH_FAILED`、`ILLEGAL_INSTRUCTION`、
`ECC_UNCORRECTABLE`……即驱动文档中记为对上下文致命的那些码）会连同它的
消息闩上 poison 标志；此后该设备上的每一次调用都会带着那条消息快速失败，
与 AMD 上一样。

---

## 内存

一个 `RawBuffer::Cuda` 携带一个设备指针、一个可选的宿主指针，以及它的
`CudaMemory` 种类，后者依据 `BufferSpec` 选出：

| `BufferSpec` | 种类 | 驱动调用 |
|---|---|---|
| 默认 | `Device` | `cuMemAlloc`——设备内存，没有宿主映射 |
| `cpu_access` | 若设备报告支持并发的托管访问则为 `Managed`，否则为 `Pinned`（WDDM、Pascal 之前） | `cuMemAllocManaged`，一个地址在两侧都有效 |
| `host` | `Pinned` | `cuMemHostAlloc(PORTABLE \| DEVICEMAP)`，内核经由总线读取它 |

`supports_device_local()` 为 `true`，因此中间结果留在设备上。
宿主 <-> 设备的复制先排空上下文（`cuCtxSynchronize`：宿主访问并不与 plan
的那些流相互定序），然后用复制流上的 `cuMemcpyHtoDAsync` /
`cuMemcpyDtoHAsync`、以 4 MiB 为块、通过一个惰性分配的**固定（pinned）暂存
缓冲区**搬运数据，每块同步一次该流。固定缓冲区则直接 `memcpy`。
设备到设备的 `_transfer` 是 `cuMemcpyDtoDAsync`；一次分配内部相互重叠的
范围会经由一个临时缓冲区中转，以保持 `memmove` 语义。释放会先排空；
若排空失败（上下文已被毒化），该分配会被**隔离**（泄漏），而不是在一个
仍在飞行中的内核之下被释放。与每个计算分配器一样，它坐落在 `LruAllocator`
之下。

---

## 程序与启动

`CudaProgram::load` 把 PTX 文本连同 16 KiB 的错误与信息日志缓冲区交给
`cuModuleLoadDataEx`，因此一次 JIT 失败会浮现为
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
消息中带上寄存器与共享内存的数字。

`Program::execute` 在设备的调度流上启动，并可选地在其上等待；
`execute_timed` 在启动前后记录一对计时 event 并返回 `cuEventElapsedTime`，
因此 BEAM 按 GPU 时间对候选排名。

---

## plan 上下文、令牌、timeline

每个执行 plan 得到一个 `CudaPlanCtx`：**一个非阻塞流**，那就是它的车道。
`dispatch` 在其上启动；设置了 `profile` 时，它会用计时 event 把这次启动
括起来，并返回一个 `CudaDispatchTimestamps`（[剖析](./profiling.md)）。
`completion_token` 记录一个仅完成用的 event（`CU_EVENT_DISABLE_TIMING`），
它的 `wait` 是 `cuEventSynchronize`，`retired` 是 `cuEventQuery`；
`synchronize` 是 `cuStreamSynchronize`。

执行器的跨 plan 定序使用 `CudaTimelineSignal`，一条由 event 发布的
timeline：`signal(stream, value)` 在流的尾部记录一个 event 并归档
`(value, event)`；`value()` 把每一次已退休的发布折叠进下界；`wait(target)`
在携带最小的 `>= target` 值的那个 event 上阻塞；`wait_on_stream` 用
`cuStreamWaitEvent` 把 GPU 工作排在它之后。槽位是 `Arc` 的，因此一个等待者
可以在另一个线程把它折叠掉的同时让自己的 event 保持存活。

---

## 图

`CudaGraph::capture` 把一条被捕获的内核链变成一张真正的 **CUDA 图**：
每个内核一个 `cuGraphAddKernelNode_v2`，其依赖列表恰好就是
`GraphKernel::deps`，即宿主侧的冒险分析。因此相互独立的内核可以在设备上
重叠执行（AMD 后端丢弃 `deps`，因为单条顺序环让它们变得多余）。每个节点的
params 经由与即时启动相同的 `extra` 协议指向该内核的 kernarg blob；图用
`cuGraphInstantiateWithFlags` 实例化。对于空链、非 CUDA 程序，或另一个设备的
程序，捕获会谢绝（`Ok(None)`）。

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
backend:             nvptx-clang
target_architecture: nvptx64-nvidia-cuda/sm_86
toolchain:           <clang identity>
flags:               -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 -Wno-override-module - -o -
abi:                 ptx-kernel-abi-v1;warp-size=32
object_format:       ptx-text-v1
```

缓存存的是 **PTX 文本**，绝不是 cubin：驱动在加载时汇编它，并把 SASS 留在
自己的 `~/.nv/ComputeCache` 里。每一次缓存命中在抵达驱动之前都会被重新校验
（`validate_ptx`），见[代码生成](./codegen.md)。`SVOD_OBJECT_CACHE=0` 关闭该
缓存，`SVOD_OBJECT_CACHE_DIR` 则可迁移它的位置。

设备工厂（`create_cuda_device`）还会拒绝一台每 block 共享内存上限低于优化器
profile 静态 `shared_max` 的设备，否则一个按该 profile 定尺的内核就只会在
JIT 时才失败。
