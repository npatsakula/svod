---
sidebar_label: Metal
---

# Metal 后端

Svod 通过 Metal 在 Apple GPU 上运行。该后端直接针对 Objective-C 运行时编写：
`libobjc`、`Metal.framework` 和私有的 `MTLCompiler.framework` 在运行时被
`dlopen`，每次调用都是一个带有手工声明 C 签名的 `objc_msgSend`
（`device/src/metal/objc.rs`）。没有 `objc2` 或 `metal` crate，没有 cargo feature，
也没有 `cfg(target_os)` 门控：该模块在每台宿主上都能编译并通过类型检查，Linux 机器
只是 `dlopen` 失败，从不注册该设备。内核由 C 渲染器的 Metal 方言
（`codegen/src/c/metal.rs`）渲染为 Metal Shading Language，并在进程内编译为
metallib。

代码位于 `device/src/metal/`（设备、分配器、编译、程序、graph、Metal 4 性能
分析器）、`runtime/src/devices/metal.rs`（设备工厂）以及
`codegen/src/c/metal.rs`（方言）。

---

## 状态

该后端于 2026 年 9 月合入，并已在一个硬件系列上得到实际运行：macOS 26 下的
**Apple9**（M3/M4 级别），在其上张量测试套件、ONNX 测试套件和 `tk` 硬件测试全部
通过，flash attention 和 GEMM 通过 `simdgroup_matrix` 运行。为旧系统编写的路径
（公开的 `newLibraryWithSource:` 编译回退、Apple9 之前的间接命令缓冲区变通方案、
`metal3.x` / `metal2.0` 语言标准）已实现，但尚未在此类硬件上验证。没有 Metal 设备
时硬件测试会自动跳过，因此 Linux CI 只运行宿主侧测试。

只支持系统默认设备（`METAL:0`）；`MTLCopyAllDevices` 枚举是后续工作
（`device/src/metal/device.rs`）。

---

## 选择设备

`METAL[:N]` 是唯一的写法（AMD 和 CUDA 有 `HIP` 式别名，Metal 没有）。在 macOS 上它
也是**平台默认值**：没有任何东西选择设备时，`default_device()` 解析为 `METAL:0`，
因此在 Mac 上通过 `SVOD_DEVICE=CPU` 重新选用 CPU 后端
（`dtype/src/default_device.rs`）。

```bash
SVOD_DEVICE=METAL:0 cargo run --release -p svod-model --example gigaam_infer -- ./audio.wav
```

`svod_device::metal::has_devices()` 加载 Objective-C 运行时并调用
`MTLCreateSystemDefaultDevice`；只有成功时，运行时的设备注册表才会注册 `"METAL"`
工厂。打开设备时会记录一行包含设备名和 GPU 系列的 `info` 日志
（`RUST_LOG=svod_device=info`）。

GPU 系列通过 `supportsFamily:` 从 Apple12 向下探测到 Apple1，再探测 Mac2，并保存为
`MetalFamily { Unknown, Mac2, Apple(n) }`。它是渲染器的 `gpu_arch`，作为目标缓存的
键，并选择优化器配置（`OptimizerRenderer::for_metal_family`）：`simdgroup_matrix`
tensor core 需要 Apple7 或更新的系列。

---

## 代码生成：MSL 方言

`CRenderer::metal()` 就是带 `CDialect::Metal` 的 CPU C 渲染器；它的存在不改变 Clang
的输出。一个内核渲染为

```c
#include <metal_stdlib>
using namespace metal;

kernel void r_64_32(device float* data0, device float* data1, constant int& data2,
                    uint3 gid [[threadgroup_position_in_grid]],
                    uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup __attribute__((aligned(16))) float local0[32];
  ...
}
```

| 概念 | Clang 方言 | Metal 方言 |
|---|---|---|
| 缓冲区参数 | `float* restrict data0` | `device float* data0` |
| 标量参数 | `const int data2` | `constant int& data2` |
| 启动 ID | `core_id` 变量 | `gid.xyz`（`gidx*` / `idx*`）、`lid.xyz`（`lidx*`），追加在 PARAM 列表之后 |
| local 缓冲区 | 栈数组 | `threadgroup __attribute__((aligned(16))) T localN[size]` |
| 屏障 | 无 | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| 地址空间 | 无 | 指针转换上的 `device` / `threadgroup` / `thread` |
| 16 位浮点 | `_Float16` | `half`、`bfloat` |
| 位转换 | union / memcpy | `as_type<T>()` |

没有 `[[buffer(n)]]` 属性：Metal **按位置**绑定参数，因此参数的绑定索引就是它在
签名中的位置，加载器也照此对应（见下文）。最多存在三个网格轴；调度器会折叠更多的
全局轴（Metal 优化器配置中的 `global_max`）。

**类型。** Float64、所有 fp8 格式以及宽于 4 的向量在渲染时被拒绝
（`reject_unsupported_metal_dtypes`，`codegen/src/c/types.rs`）；调度器事先把内部的
f64 降为 f32。bf16 算术经由 `float` 提升，bf16 收窄使用整数就近舍入到偶数的模式集。

**数学函数。** `sqrt`、`exp2` 和 `log2` 是原生的；`sin` 渲染为 `precise::sin`；
`exp`、`log`、`cos`、`tan` 和 `erf`（MSL 没有 `erf`）由共享的
`amd_decomposition_patterns()` 基于原生 `exp2`/`log2` 分解，与 AMD 上相同。渲染器的
`extra_matcher` 是 CPU 的那个（`cpu_extra_matcher()`）。快速数学在所有地方都关闭
（`-fno-fast-math`，公开路径上为 `MTLMathModeSafe`），以保证共享的测试容差成立。

**Tensor core。** `Wmma` 降级为基于 `simdgroup_<T>8x8` 和
`simdgroup_multiply_accumulate` 的逐形状辅助函数：只有一种形状，在 32 个线程上做
8×8×8，每个通道两个元素，支持 f32→f32、f16→f32、f16→f16、bf16→f32 和 bf16→bf16
（优化器配置中的 `METAL_888`）。`tk` 增加了 `simd_shuffle`、
`simd_shuffle_xor` 和 `simdgroup_barrier` 构建器（`codegen/src/c/metal.rs`），
Apple 的 flash-attention 和 GEMM 内核正是由它们构建的。

---

## 编译路径

`compile_msl`（`device/src/metal/compile.rs`）把源码发送给 Apple 私有的
`MTLCodeGenService`——与 tinygrad 使用的路径相同——并通过一个手工构建的 Objective-C
block 回调接收 metallib（`MTLB` 魔数，`ENDT` 尾部），超时 60 秒，每次只处理一个
请求。标志为

```text
-fno-fast-math -std=<std> --driver-mode=metal -x metal -fno-caret-diagnostics
-fmodules-cache-path=<cache>/metal-modules
```

其中 `<std>` 随 macOS 主版本而定（26+ 上为 `metal4.0`，14–25 上为 `metal3.1`，
13 上为 `metal3.0`，更早为 `macos-metal2.0`），模块缓存把 `metal_stdlib` 的解析
从约 250 ms 降到约 8 ms。

`MTLCompiler.framework` 以 `RTLD_GLOBAL` 加载自己的 libLLVM，它无法与 CPU 后端的
进程内 libLLVM 共存，因此两者争用一个槽位（`claim_inprocess_llvm`）。竞争的失败方，
或没有该私有框架的系统，会采用 `compile_msl_public`：执行一次
`newLibraryWithSource:options:error:` 编译以便暴露诊断信息，此后 **MSL 源码本身**
就是载荷，程序加载器在加载时再编译一次。两种载荷共享同一个目标缓存条目：

```text
backend:             metal
target_architecture: Apple9/air64
toolchain:           macos=26.0
flags:               -fno-fast-math -std=metal4.0 --driver-mode=metal -x metal -fno-caret-diagnostics
abi:                 msl-kernel-abi-v1
object_format:       metallib-or-msl-v1
```

传输方式被刻意排除在身份之外：赢得 libLLVM 槽位的 BEAM 工作进程与输掉槽位的父
进程必须在键上达成一致。

---

## 程序与启动

`MetalProgram::load` 接受任一种载荷（metallib 用 `newLibraryWithData:`，MSL 用
`newLibraryWithSource:`），用 `newFunctionWithName:` 绑定函数，并以
`setSupportIndirectCommandBuffers:YES` 构建管线，读取
`maxTotalThreadsPerThreadgroup`、`threadExecutionWidth` 和
`staticThreadgroupMemoryLength`。

参数按位置绑定：缓冲区用 `setBuffer:offset:atIndex:`——宿主指针通过设备的
`PointerRegistry`（一个以缓冲区基址为键的 `BTreeMap`）解析为其
`(MTLBuffer, offset)`——标量用 `setBytes` 作为 4 字节 `i32`；超出 `i32` 范围的值是
运行时错误。ABI 槽位必须递增，可以有空缺，最多 31 个绑定（`MAX_BUFFER_BINDINGS`）。
`global_size` 是线程组数，`local_size` 是每组线程数，经由
`dispatchThreadgroups:threadsPerThreadgroup:` 发送；没有 local 轴的内核每组运行一个
线程，超过 `maxTotalThreadsPerThreadgroup` 的组会被拒绝。

每次派发是一个以内核名标注的命令缓冲区，提交到一个深度为 1024 的单一队列。当
`wait = false` 时它加入设备的 `in_flight` 列表；`MetalDevice::synchronize` 用
`waitUntilCompleted` 等待每个条目，并报告第一个 `NSError`。宿主访问缓冲区前会先
排空设备。`execute_timed` 读取命令缓冲区的 `GPUStartTime` / `GPUEndTime`，BEAM 正是
据此对候选排序。

---

## 内存

每次分配都是一个 `MTLResourceStorageModeShared` 模式的 `MTLBuffer`：Apple
silicon 是统一内存，因此 `BufferSpec` 标志被忽略，`copyin` / `copyout` /
`_transfer` 在 `synchronize()` 之后对 `contents` 执行宿主 `memcpy` / `memmove`。
没有 private 或 managed 缓冲区，也没有 blit 编码器。释放前会先排空设备；若排空
失败，该分配会被泄漏，而不是在仍有内核运行时被释放。

---

## Graph

`MetalGraph::capture` 把一条内核链记录进一个由 `ConcurrentDispatch` 命令组成的
`MTLIndirectCommandBuffer`，每条命令都带 `setBarrier`，从而保持捕获顺序。一次重放
就是一个命令缓冲区：先对绑定的缓冲区调用 `useResources:count:usage:`，再调用
`executeCommandsInBuffer:withRange:`。`replay` 等待上一次重放，并只重新绑定缓冲区
发生变化的槽位。以下情况捕获会放弃（返回 `Ok(None)`，改为逐次派发）：空链、不是
`MetalProgram` 的程序、名称中包含 "virtual" 的设备（半虚拟化的 CI GPU 会破坏
ICB）、超过 32 位的偏移，或**任何标量参数**——带符号形状的链不会被 graph 化。在
Apple9 以下会应用 tinygrad 的 `FIX_METAL_ICB` 变通方案（每个管线一次空派发）。

---

## 性能分析

| 层级 | Metal 上 | 来源 |
|---|---|---|
| 1 — 设备时间 | 是 | 每个命令缓冲区的 `GPUStartTime` / `GPUEndTime`；在 graph 内部，使用 Metal 4 计数器堆（macOS 26+）或每个内核一个命令缓冲区 |
| 2 — roofline | 是 | 与后端无关 |
| 3 — 静态资源 | 部分 | `lds_bytes` 来自 `staticThreadgroupMemoryLength`，`wave_size` 来自 `threadExecutionWidth`，`occupancy` 为 `maxTotalThreadsPerThreadgroup / 1024`；没有寄存器计数 |
| 4 — 硬件计数器 | 否 | |

`Mtl4Profiler`（`device/src/metal/mtl4.rs`）只用于带性能分析的 graph 重放：它在
Metal 4 命令缓冲区中、于两个精确时间戳之间单独运行每条间接命令，对绑定的缓冲区
使用驻留集，并等待一个共享事件。MTL4 编码器会静默跳过其管线未事先提供给它的间接
命令的首次执行，因此每个管线都会先设置到编码器上。

---

## 限制

- 一个设备（`METAL:0`）、一个命令队列、只有共享存储；
- 标量为 `i32`；没有 f64、没有 fp8、没有宽于 4 的向量；
- 只有一种 tensor core 形状（`simdgroup` 8×8×8）；
- graph 不包括带标量参数的链；
- 没有硬件计数器、没有寄存器计数，graph 之外的逐次派发计时是整个命令缓冲区的
  时间戳；
- 快速路径是 Apple 未公开文档的 `MTLCodeGenService`；公开 API 是回退方案。

没有 Metal 专属的环境变量。适用的是共享变量：`SVOD_DEVICE`、
`SVOD_OBJECT_CACHE` / `SVOD_OBJECT_CACHE_DIR`、用于模块缓存的 `XDG_CACHE_HOME`，
以及用于 graph 捕获与放弃信息的 `RUST_LOG=svod_device=debug`。

---

## 测试

```bash
cargo test -p svod-device metal          # host tests everywhere; hardware tests self-skip
cargo test -p svod-codegen metal         # MSL golden tests
SVOD_DEVICE=METAL:0 cargo test -p svod-tensor   # codegen_tests! `metal` variants
SVOD_DEVICE=METAL:0 cargo test -p svod-onnx
```
