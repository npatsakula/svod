---
sidebar_label: 概览
---

# 后端

后端是渲染后内核之下的一切：把 UOp IR 变成源码的渲染器、把源码变成目标文件的
编译器、把目标文件变成可调用 `Program` 的加载器、一个分配器，以及可选的
graph。Svod 提供四个后端，全部位于同一个二进制文件中；给定宿主上具体存在哪些，
在运行时决定。

| 设备 | 硬件 | 渲染器 | 编译路径 | Graph 重放 | 状态 |
|---|---|---|---|---|---|
| [`CPU`](./cpu.md) | x86_64、aarch64、riscv64、loongarch64、ppc64le | LLVM IR 文本（默认）或 C | 进程内 libLLVM，否则 `clang -c`；[内存内 ELF 加载器](./jit-loader.md) | 无（同步调用） | 生产可用 |
| [`AMD:N`](./amd/overview.md) | CDNA3、RDNA3、RDNA3.5、RDNA4（Linux，KFD） | LLVM IR 文本，AMDGPU 目标 | 以 `amdgcn` 目标调用 `clang` → 加载进 VRAM 的 ELF 代码对象 | AQL 命令流（PM4 需显式启用） | 生产可用 |
| [`CUDA:N`](./cuda/overview.md) | NVIDIA，驱动 CUDA 12.0+ | LLVM IR 文本，NVPTX 目标 | 以 NVPTX 目标调用 `clang` → PTX，已安装时用 `ptxas`，否则用驱动 JIT | CUDA graphs | 生产可用 |
| [`METAL:N`](./metal.md) | Apple GPU | C，Metal 方言 | 进程内的 Apple `MTLCodeGenService` → metallib | 间接命令缓冲区 | 已在 Apple9 / macOS 26 上验证 |

每个后端编译出的目标文件都经过同一个磁盘目标缓存，其键由源码和每个后端各自的
`CompilerIdentity` 组成（[CPU 页面](./cpu.md)）。

---

## 选择设备

`SVOD_DEVICE` 选择张量与内核的默认设备。其值按 `NAME[:N]` 不区分大小写地解析
（`dtype/src/default_device.rs`）：

| 值 | 设备 |
|---|---|
| `CPU` | `DeviceSpec::Cpu` |
| `AMD[:N]`、`HIP[:N]` | `DeviceSpec::Amd { device_id }`——KFD 拓扑中的第 N 个 GPU 节点 |
| `CUDA[:N]`、`GPU[:N]` | `DeviceSpec::Cuda { device_id }` |
| `METAL[:N]` | `DeviceSpec::Metal { device_id }`（只有 `0` 存在） |

只写 `NAME` 即设备 0。`NV` 被刻意拒绝——这个名字保留给未来的用户态 NVIDIA 驱动。
没有任何东西选择设备时，采用平台默认值：**macOS 上为 `METAL:0`，其他地方为
`CPU`**。完整的优先级依次是：`with_default_device` 作用域、线程局部的
`set_default_device`、`SVOD_DEVICE`（每个进程读取一次）、平台默认值。GPU 架构
从不是 spec 的一部分：它是已打开设备的属性，因此一块物理 GPU 只有一个身份，
内核缓存以设备报告的内容为键。

`svod-device` 中的 `DeviceSpecExt::parse` 接受同样的写法，另外还有
`DISK:<path>`（一个只读、内存映射的文件设备，不能运行内核）和 `WEBGPU`，
后者尚无分配器，会以 `DeviceUnavailable` 失败。

---

## 运行时检测的注册

每个后端都在每台宿主上编译——AMD、CUDA 或 Metal 都没有 cargo feature。CUDA 和
Metal 绑定是基于 `libloading` 的纯 Rust 代码，可在任何地方编译；面向 AMD 内核
驱动的模块是 `cfg(unix)`。因此 Linux 或 macOS 上的 `cargo check` 会对它们全部做
类型检查。后端是否*可用*，
在设备工厂注册表首次被访问时决定
（`runtime/src/device_registry.rs`）：

```rust
registry.register_factory("CPU", ...);                        // always
if svod_device::amd::has_devices()   { registry.register_factory("AMD", ...); }
if svod_device::metal::has_devices() { registry.register_factory("METAL", ...); }
if svod_device::cuda::has_devices()  { registry.register_factory("CUDA", ...); }
```

每个探测都无副作用且带记忆化：AMD 读取 KFD sysfs 拓扑并询问是否有节点属于受支持
的架构；Metal 对 Apple 框架执行 `dlopen` 并请求系统默认设备；CUDA 加载
`libcuda.so.1`，绑定它用到的每个入口点，调用 `cuInit` 并统计设备数。没有对应硬件
的宿主就根本没有这种设备类型，请求它会以 `UnsupportedDevice` 失败。到处编译一切
的意义在于：对共享的 `Program` / `PlanContext` / `Graph` trait 的改动会在任何
开发者机器上打破构建，而不仅仅是在装有 GPU 的那台上。

注册表为每个 `DeviceSpec` 缓存一个 `Device`（`DEVICE_FACTORIES`）；构造过程——
打开 KFD、探测工具链——在映射锁之外运行，按 spec 串行化，构造失败会让槽位保持
为空以便重试。分配器位于 `svod-device` 中一个独立的注册表
（`registry::registry()`），其中每个计算分配器都被包装在 `LruAllocator` 中，
按大小和 spec 池化已释放的缓冲区。

---

## 后端需要实现什么

一个 `Device`（`device/src/device.rs`）由五部分组成：

```rust
pub struct Device {
    pub device: DeviceSpec,
    pub allocator: Arc<dyn Allocator>,
    pub compilers: Vec<CompilerPair>,     // (Arc<dyn Renderer>, Arc<dyn Compiler>)
    pub renderer: Arc<dyn Renderer>,
    pub compiler: Arc<dyn Compiler>,
    pub runtime: RuntimeFactory,          // Fn(&CompiledSpec) -> Result<Box<dyn Program>>
    pub graph: Option<GraphFactory>,      // Fn(&[GraphKernel]) -> Result<Option<Box<dyn Graph>>>
}
```

| Trait | 必需方法 | 作用 |
|---|---|---|
| `Renderer` | `render`、`device`、`supported_ops` | UOp 图 → `ProgramSpec`（源码、入口、ABI、启动尺寸）。`gpu_arch` 选择优化器配置；`decompositor` 和 `extra_matcher` 降级目标无法选择的操作 |
| `Compiler` | `compile`、`cache_key` | `ProgramSpec` → `CompiledSpec` 字节；`cache_key` 是作为目标缓存键的 `CompilerIdentity` |
| `RuntimeFactory` | — | 把 `CompiledSpec` 加载为 `Program`；`Device::new` 对它进行包装，使每个 spec 的阶段身份先被校验 |
| `Program` | `execute`、`name` | 一次内核启动；`execute_timed`（供 BEAM 使用的 GPU 时钟时长）、`new_exec_context`、`resource_usage` 和 `as_any` 是可选的 |
| `PlanContext` | `dispatch`、`synchronize` | 由 `Program::new_exec_context` 创建的每个计划的状态：通道、完成令牌、时间戳、计数器（`set_pmc`）、原生链接重放（`replay_linked_plan`） |
| `Allocator` | `_alloc`、`name`、`device_spec` | `_copyin` / `_copyout` / `_transfer` / `_free` / `synchronize` / `supports_device_local` 是可选的，默认采用宿主内存语义 |
| `Graph` | `replay` | 以一次提交重放的已捕获内核链；`completion_token`、`replay_profiled` 可选 |
| `CompletionToken`、`TimelineSignal`、`DispatchTimestamps` | | 执行器消费的同步与性能分析句柄（`device/src/sync.rs`） |

启动约定由所有 GPU 后端共享：`global_size` 是以工作组为单位的网格，`local_size`
是以线程为单位的工作组；CPU 把 `global_size[0]` 用作 `core_id` 切分。内核参数是
ABI 的 `PARAM` 槽位，按顺序排列——先指针，后 `i32` 标量——每个加载器都以同样方式
打包（AMD 和 CUDA 上为 `ClikeKernargLayout`，Metal 上为按位置的
`setBuffer`/`setBytes`，CPU 上为 libffi CIF）。

### 添加后端

`runtime/src/devices/` 中的四个工厂就是模板。每个 `create_*_device` 都做同样的
五件事：

1. 从注册表获取其 `DeviceSpec` 对应的分配器；
2. 围绕一个代码生成入口点构建渲染器包装
   （`LlvmTextRenderer::amd(arch)`、`LlvmTextRenderer::nvptx(arch)`、
   `CRenderer::metal()` 或 CPU 渲染器），声明 `supported_ops`、
   分解模式和 `gpu_arch`；
3. 构建带有 `CompilerIdentity` 和 `ObjectCache` 的编译器，产出加载器能够校验的
   字节（`validate_amd_object`、`validate_ptx` /
   `validate_cubin`、`validate_metallib`，CPU 上为 ELF 检查）；
4. 安装一个 `RuntimeFactory`，把这些字节加载为该后端的
   `Program`；
5. 可选地调用 `with_graph(...)` 以支持捕获/重放。

然后它以自己的设备类型字符串注册工厂，并以 `has_devices()` 探测作为门控；调度器
还需要为新目标提供一份优化器配置（`OptimizerRenderer::for_*`：wave 大小、
tensor core 形状、共享内存与 local 限制）。每个后端都另有一个 `create_*_codegen`，
以便 BEAM 工作进程无需打开设备即可渲染和编译。
