---
sidebar_label: 调试
---

# 调试

后端有三个失败面：宿主工具链（clang、对象缓存）、驱动 JIT
（`cuModuleLoadDataEx`），以及运行时的设备（驱动错误、内存故障）。本页列出
各个旋钮，以及如何读懂每一类错误。

---

## 环境变量

| 变量 | 默认 | 效果 |
|---|---|---|
| `SVOD_DEVICE` | `CPU` | `CUDA:N`（别名 `NV`、`GPU`）选择默认的张量设备 |
| `SVOD_DUMP_NVPTX_IR` | 未设置 | 接收每个内核 NVPTX LLVM IR 的目录，文件名为 `sm_XY_<kernel>.ll` |
| `SVOD_OBJECT_CACHE` | 开 | `0` 关闭磁盘上的 PTX 缓存 |
| `SVOD_OBJECT_CACHE_DIR` | `$XDG_CACHE_HOME` / `~/.cache` | 迁移缓存位置 |
| `SVOD_PROFILE_ITERS`、`SVOD_ORIGIN`、`SVOD_ORIGIN_DEPTH` | | profiler 旋钮，见[剖析](./profiling.md) |
| `RUST_LOG` | 未设置 | `svod_device=debug` 显示设备打开行、PTX JIT 信息日志、图捕获与重放回退；`svod_runtime=debug` 显示每一次 clang 调用 |

没有 CUDA 专属的调度转储；驱动 JIT 日志与 `tracing` 覆盖了
`SVOD_DEBUG_DISPATCH` 在 AMD 上所做的事。

---

## 一台明明有设备的宿主上出现 "No CUDA device"

当库加载不了、某个被绑定的符号缺失、`cuInit` 失败或者计数为零时，
`has_devices()` 为 `false`。`CudaDevice::open` 会说明是哪一种：

```text
device unavailable: cannot load libcuda.so.1: ...        # no driver on the loader path
device unavailable: libcuda.so.1 has no symbol cu...     # driver too old for a bound entry point
no CUDA GPU available: CUDA cuInit failed: ...           # driver loaded, no usable device
```

检查 `ldconfig -p | grep libcuda`、`nvidia-smi`，以及该进程能否打开
`/dev/nvidia*`。

---

## 读懂一次 JIT 失败

一个被驱动拒绝的 PTX 会浮现为 `Error::CudaJit`，它的显示形式是原因后面跟着
驱动的错误日志：

```text
CUDA JIT of kernel "r_64_32" failed: CUDA_ERROR_INVALID_PTX (218): a PTX JIT compilation failed
ptxas application ptx input, line 27; error   : ...
```

`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` 意味着驱动比模块中的 PTX ISA 更老
（锁定在 7.8，CUDA 11.8 / R520），见[要求](./overview.md)。信息日志
（警告、寄存器溢出）以 `debug` 级别记录在 `svod_device` 之下。

有两个错误来自 Svod 自己的校验器，在驱动看到任何东西之前：

```text
PTX references an unresolved function: .extern .func ...   # an LLVM intrinsic name the NVPTX
                                                            # backend did not recognize
cached PTX targets sm_80, not sm_86                          # a corrupt or foreign cache entry
```

第一个才是要紧的陷阱：一个拼错的 `llvm.nvvm.*` intrinsic 并不是 clang 错误，
它会变成一次外部调用。修复之处在 `codegen/src/llvm/nvptx/`，或者
`codegen/src/llvm/text/mod.rs` 中的 intrinsic 声明表。

---

## 用 toolkit 做离线检查

运行时没有任何东西需要 CUDA toolkit，但如果它已安装，它的工具可以作用于
转储出来的 IR：

```bash
SVOD_DUMP_NVPTX_IR=/tmp/nvptx SVOD_DEVICE=CUDA:0 cargo test -p svod-tensor -- some_test

# Reproduce the exact compile, then assemble with ptxas to see the real diagnostics
clang -x ir -S -O3 --target=nvptx64-nvidia-cuda -march=sm_86 -Wno-override-module \
      /tmp/nvptx/sm_86_r_64_32.ll -o r_64_32.ptx
ptxas -arch=sm_86 -v r_64_32.ptx -o r_64_32.cubin   # -v prints registers, shared, spills
nvdisasm r_64_32.cubin | less                         # the SASS
```

`ptxas -v` 是看清 `maxThreadsPerBlock` 为何小于启动所需的最快方式：寄存器
压力顶上了 `.maxntid` 这个 launch bound。对应的运行时错误会把数字点出来：

```text
CUDA kernel 'r_64_32' block [512, 1, 1] (512 threads) exceeds its maxThreadsPerBlock 256
  (numRegs 96, sharedSizeBytes 4096, localSizeBytes 0)
```

---

## 运行时的驱动错误

每一次驱动调用都会被检查，并用驱动自己的名称与文本报告出来：

```text
CUDA cuStreamSynchronize failed: CUDA_ERROR_ILLEGAL_ADDRESS (700): an illegal memory access was encountered
```

内核故障是异步的：`cuLaunchKernel` 成功返回，而错误落在下一次同步调用上
（`cuStreamSynchronize`、`cuEventSynchronize`、`cuCtxSynchronize`，或一次宿主
复制）。驱动文档中记为**粘性**的那些码（`ILLEGAL_ADDRESS`、
`LAUNCH_FAILED`、`ILLEGAL_INSTRUCTION`、`MISALIGNED_ADDRESS`、
`ECC_UNCORRECTABLE`……）会毒化设备：此后每一次调用都会立刻带着已记录的消息
失败，而释放会隔离它们的分配，而不是把一个挂死的内核可能仍在触碰的内存
交还回去。

与 AMD 后端不同，这里没有 VA 注册表来对故障地址分类；驱动并不暴露它。要定位
一次故障，可以在 toolkit 的 sanitizer 下跑同一个二进制，它对驱动 API 程序和
JIT 加载的 PTX 都管用：

```bash
SVOD_DEVICE=CUDA:0 compute-sanitizer --tool memcheck \
  target/release/examples/gigaam_infer ./audio.wav
```

如果不是故障而是结果错了，那么图重放的回退值得一看：当一次重放的缓冲区
别名关系与捕获时不同，`RUST_LOG=svod_device=debug` 会打印 `CUDA graph replay
with re-aliased buffers; using the capture-order chain`；而
`SVOD_DEVICE=CUDA:0 cargo test -p svod-tensor` 会运行 `codegen_tests!` 的
`cuda` 变体，也就是 CPU 后端所通过的那套张量级断言，一次一个内核。

:::tip 流水线调试器
对于编译器侧的问题（哪些 UOp 产生了哪些 IR），`/svod-debug` skill 记录了
前端 → codegen 的追踪目标；`SVOD_DUMP_NVPTX_IR` 是那一家族中 CUDA 的成员，
与 `SVOD_DUMP_AMD_IR` 和 `SVOD_DUMP_LLVM_IR` 并列。
:::
