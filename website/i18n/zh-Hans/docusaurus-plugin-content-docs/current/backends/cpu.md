---
sidebar_label: CPU
---

# CPU 后端

CPU 后端在运行时把每个内核编译为原生机器码，并在进程内调用它。它有两条代码路径，
由 `SVOD_CPU_BACKEND` 选择：

| `SVOD_CPU_BACKEND` | 渲染器 | 编译器 | 缓存后端 |
|---|---|---|---|
| `llvm`（默认；未设置或为空时选中它） | LLVM IR 文本（`codegen/src/llvm/cpu/`） | 进程内绑定的 libLLVM，无法加载时用 `clang -x ir` | `cpu-llvm-inprocess` / `cpu-llvm-clang` |
| `clang` | C 源码（`codegen/src/c/`，Clang 方言） | `clang -x c` 子进程 | `cpu-clang` |

只接受这两种写法及其大写形式（`CpuBackend::parse`）；其他任何值都会记录一条警告并
选择 LLVM。两条路径都产出可重定位的 ELF 目标文件，由 [JIT 加载器](./jit-loader.md)
在内存中映射并重定位；两者都不写临时文件，也不 `dlopen` 共享库。相关接线位于
`runtime/src/devices/cpu.rs`。

`"CPU"` 设备工厂无条件注册，CPU 在除 macOS 之外的所有地方都是默认设备，macOS 上
默认设备是 `METAL:0`（见 [后端](./overview.md)）。

---

## LLVM 路径

渲染器为每个内核发出一个 LLVM 函数，缓冲区作为 `ptr noalias align 32` 参数，
标量作为 `i32`，并带有属性
`nounwind "no-builtins" "no-trapping-math"="true"`：

```llvm
define void @r_64_32(ptr noalias align 32 %data0, ptr noalias align 32 %data1, i32 %core_id) #0 {
entry:
  ...
  ret void
}
```

### 进程内 libLLVM

`runtime/src/llvm_inprocess.rs` 通过 `libloading` 从共享 libLLVM 绑定 LLVM-C API。
该库每个进程查找一次，顺序如下：

1. `SVOD_LLVM_LIB=<path>`——此时该文件是唯一候选；
2. `llvm-config --libdir` 与每个候选名拼接；
3. 通过动态加载器自身的搜索路径查找每个候选名；
4. 在 macOS 上，`/opt/homebrew/opt/llvm/lib/libLLVM.dylib` 以及带版本的
   `llvm@N` keg。

候选名依次是 `libLLVM.so` / `libLLVM.dylib`，然后对于从 30 递减到 16 的每个主版本，
是各发行版的 SONAME（`libLLVM.so.N.1`、`libLLVM-N.so.1`、
`libLLVM-N.so`、`libLLVM.so.N`；`libLLVM-N.dylib`、`libLLVM.N.dylib`）。版本通过
`LLVMGetVersion` 读取，且必须**不低于 16**。

一次编译从内存缓冲区解析 IR（`LLVMParseIRInContext`），对其做校验，运行启用了
循环展开、循环向量化和 SLP 向量化的 `LLVMRunPasses("default<O2>")`，并用
`LLVMTargetMachineEmitToMemoryBuffer` 发出目标文件。目标机器为
`<arch>-none-unknown-elf`，使用宿主 CPU 名称和特性字符串
（`LLVMGetHostCPUName` / `LLVMGetHostCPUFeatures`）、PIC 重定位和默认代码模型；
在 macOS aarch64 上特性字符串会追加 `+reserve-x18`。上下文处理器收到的错误级别
诊断会让编译失败。每个线程在线程局部变量中持有一个 `Session`（上下文、目标机器、
数据布局、pass 选项），因此内核可以并发编译。

`SVOD_LLVM_INPROCESS=0`（精确值 `0`）禁用进程内路径。任何绑定库的失败——找不到、
版本过旧、缺少符号——都会回退到下面的 clang 生成器；除非是该变量有意禁用，否则
回退会以 `warn` 级别记录。

:::note[每个进程一个 libLLVM]
Apple 的 Metal 编译器框架以 `RTLD_GLOBAL` 加载自己的 libLLVM，它无法与另一份副本
共存于同一进程。因此 CPU 后端和 [Metal](./metal.md) 后端争用同一个槽位
（`svod_device::claim_inprocess_llvm`）：先编译的一方保有它，另一方采用其子进程
或公开 API 回退。
:::

### clang 回退

`compile_ir_to_object_with` 把 IR 通过管道交给 clang，从 stdin 到 stdout：

```text
clang -x ir -c -O2 -march=native -fPIC -fno-math-errno -fno-stack-protector \
      -funroll-loops -fvectorize -fslp-vectorize --target=<arch>-none-unknown-elf [-ffixed-x18] - -o -
```

进程内生成器与 clang 生成器带有不同的目标缓存身份（`cpu-llvm-inprocess` 带
`library=<path>;version=x.y.z` 及宿主 CPU/特性，对比 `cpu-llvm-clang` 带 clang
身份），因此目标文件绝不会在生成器之间混用。

---

## clang C 路径

`SVOD_CPU_BACKEND=clang` 改为渲染 C——`void name(T* restrict data0, ...,
const int dataN)`，数学函数用 `__builtin_*`——并以如下命令编译

```text
clang -c -x c -O2 <cpu flag> -fPIC -ffreestanding -fno-math-errno -fno-stack-protector \
      -nostdlib -fno-ident --target=<arch>-none-unknown-elf [-ffixed-x18] - -o -
```

其中 `<cpu flag>` 在 x86_64 和 loongarch64 上为 `-march=native`，在 riscv64 上为
`-march=rv64g`，其他地方为 `-mcpu=native`（在 ARM 上 `-march=native` 只设置 ISA
系列）。源码是可读的，这正是该路径的用途：无需阅读 LLVM IR 即可查看内核做了什么。

`clang` 只在 `PATH` 上解析（`ClangToolchain::discover`）；没有 `SVOD_CLANG` 或 `CC`
覆盖。它在目标缓存中的身份是
`path=...;sha256=<binary digest>;version=<clang --version>`，C 路径还额外记录从
`clang -###` 解析出的 `-target-cpu` 与特性（当某个标志为 `native` 时用
`/proc/cpuinfo` 做指纹），因此在一台机器上编译的目标文件不会在另一种 CPU 上被复用。

:::tip[dlopen 回退]
`svod-runtime` 的 `dlopen-fallback` cargo feature **仅对 C 路径**替换 ELF 加载器：
`clang -shared ... -lm` 把 `kernel.so` 写入临时目录，再由 `libloading` 打开。它更慢，
是为内存内加载器无法工作的平台准备的；目标缓存将其记录为 `elf-shared-dlopen-v1`，
CI 以 `test-dlopen-fallback` 检查运行它。LLVM 路径始终使用内存内加载器。
:::

---

## 数学函数

CPU 渲染器保留超越函数：`exp`、`log`、`sin`、`cos`、`tan`、`pow` 等渲染为
`@llvm.*` 内建函数或 `__builtin_*` 调用。`sqrt`、`fma`、`floor`、`rint` 之类在宿主
ISA 具备相应指令时降级为指令；其余的变为 `libm` 调用，由加载器在加载时通过
`dlsym(RTLD_DEFAULT)` 解析。
LLVM 路径从其支持的操作中移除了 `Erf`（它会被分解），C 路径则保留；两者都去掉了
`Threefry` 和 `Max`，它们分别分解为一个单纯的 XOR 和一个 select。

---

## 线程

`SVOD_THREADS`（正整数；默认：宿主可用的并行度）是唯一的线程预算。它决定 rayon
全局线程池的大小——该池并行编译内核缓存未命中项并运行 CPU 内核——也是每个 CPU
内核默认的 **`core_id` 切分**：

- 优化器的 THREAD opt（`schedule/src/optimizer/opts.rs`）把一个全局循环轴移到
  最外层并切分它；启发式依次尝试 32、16、12、8、6、5、4、3 和 2 块，受预算以及
  每 131072 个元素一块的上限约束，选取第一个能整除某轴的数量（否则填充一个轴）。
  该切分被固化进内核及其缓存身份。
- `gpudims` 把该轴降级为取值 `[0, N-1]` 的 `core_id` 变量，而不是
  `gidx`/`lidx`，`global_size[0]` 变为 `N`。
- 运行时 `execute_kernel` 看到 `global_size[0] > 1`，便运行
  `(0..N).into_par_iter()`，用同样的缓冲区调用同一个函数指针，并为每个任务改写
  `core_id`。每个 `core_id` 写入互不相交的输出区间，因此无需同步。在已有的 rayon
  工作线程内部，该循环串行执行而不是嵌套。

内核切分数与线程池大小可以不同：切分为 8 份的内核在 4 个线程上也能正确运行。
`SVOD_THREADS=1` 禁用切分。Rayon 只构建一次全局线程池，因此第一个调用者的大小
生效，之后不同的请求会记录一条警告（`ensure_thread_pool`）。不会读取
`RAYON_NUM_THREADS`。

---

## 调用内核

已加载的内核是 `JitKernel`（C 路径，以 `ClangKernel` 重新导出）或
`LlvmKernel`：映射后的目标文件、入口指针、变量名以及一个 `KernelCif`。ABI 为按
ABI 槽位顺序排列的 `void kernel(ptr..., i32...)`：存储参数为 `Type::pointer()`，
标量为 `Type::i32()`，执行器携带的 `i64` 值被截断为 `i32`。调用本身经由
**libffi**（`cif.call(CodePtr(fn_ptr), &args)`，`runtime/src/dispatch.rs`），
这就是内核签名可以逐内核变化、而无需为每种参数个数做一次 transmute 的原因。

CPU 没有 graph 工厂、没有计划上下文，也没有时间戳：每个内核都是一次同步调用，
`wait` 被忽略，性能分析器的设备时间层级为挂钟时间。

---

## 目标缓存

编译出的目标文件缓存在磁盘上（`runtime/src/object_cache.rs`），位于
`SVOD_OBJECT_CACHE_DIR`，否则 `$XDG_CACHE_HOME/svod/objects`，否则
`~/.cache/svod/objects`；`SVOD_OBJECT_CACHE=0` 禁用它，
`SVOD_OBJECT_CACHE_MAX_BYTES` 设置预算（默认 1 GiB，按 mtime 最旧优先淘汰）。
每个条目以源码的 SHA-256 和一个 `CompilerIdentity`（`schema`、`backend`、
`target_architecture`、`toolchain`、`flags`、`abi`、`object_format`）为键，存储为
`<hex>.obj`，包含魔数、键、载荷摘要和载荷；完整性检查失败视为未命中。每个目标
文件，无论来自缓存还是新编译，在加载前都要针对宿主 ELF 架构、字节序、目标类型和
入口符号进行校验。同一个缓存也以各自的身份保存 AMD、CUDA 和 Metal 的目标文件。

---

## 环境变量

| 变量 | 默认值 | 作用 |
|---|---|---|
| `SVOD_CPU_BACKEND` | `llvm` | `clang` 选择 C 路径 |
| `SVOD_LLVM_INPROCESS` | 开启 | `0` 强制使用 `clang -x ir` 子进程 |
| `SVOD_LLVM_LIB` | 未设置 | 要绑定的 libLLVM 路径；设置后是唯一候选 |
| `SVOD_THREADS` | 宿主并行度 | 线程预算和默认的 `core_id` 切分 |
| `SVOD_OBJECT_CACHE` | 开启 | `0` 禁用磁盘目标缓存 |
| `SVOD_OBJECT_CACHE_DIR` | `$XDG_CACHE_HOME/svod/objects` | 迁移缓存位置 |
| `SVOD_OBJECT_CACHE_MAX_BYTES` | 1 GiB | 缓存预算 |
| `SVOD_DUMP_LLVM_IR` | 未设置 | 接收每个内核渲染出的 LLVM IR 的目录 |
| `SVOD_DUMP_POST_O2_IR` | 未设置 | 接收 clang `-O2` 流水线之后 IR 的目录 |
| `RUST_LOG` | 未设置 | `svod_runtime=debug` 会说明由哪个生成器编译（libLLVM 或 clang），并记录每次内核编译与加载；意外回退到 clang 时为 `warn` |
