---
sidebar_label: 限制与路线图
---

# 限制与路线图

后端目前还做不到什么、源码中的具体原因，以及计划做什么。这里没有任何东西
会静默失败：每一处缺口要么是一个干净的错误，要么是一条有记录的回退路径。

---

## 尚未实现

| 缺口 | 今天的状况 | 位置 |
|---|---|---|
| **fp8 转换** | 一次到 `FP8E4M3` / `FP8E5M2` 的 cast（或从它们出发的 cast）会在渲染时失败（`NVPTX fp8 cast ...`）；sm_89 的 `cvt.*.e4m3x2` intrinsic 并未发射。fp8 的 `mma.sync` 行在 `resolve_mma` 中存在，但喂不进去。 | `codegen/src/llvm/nvptx/ops.rs` |
| **带作用域的同步** | 宿主的读与写会排空整个上下文（`_copyin` / `_copyout` 中的 `cuCtxSynchronize`），而不是只等待该缓冲区的生产者。plan 与图确实会交出基于 event 的 `CompletionToken`。 | `device/src/cuda/allocator.rs` |
| **点对点复制** | `cuMemcpyPeerAsync` / `cuDeviceCanAccessPeer` 未做绑定。一次 `CUDA:0 → CUDA:1` 的复制在执行器中走 `SyncStrategy::PeerToPeer`，而它回落到 `Buffer::copy_from`；两个分配器就是两台设备，因此字节要经由一个宿主 `Vec` 中转。 | `runtime/src/executor.rs`、`device/src/buffer.rs` |
| **硬件计数器（第 4 层）** | 没有 CUPTI；`pmc_available()` 为 `false`，`SVOD_PMC` 带着一条提示退化。 | [剖析](./profiling.md) |
| **Tile 内核（`tk`）** | 仅 AMD：`resolve_arch` 对一个 CUDA spec 给不出 `AmdArch`，因此一次 `tk` 启动会报告 `UnsupportedArch`。 | `tk/src/target.rs` |
| **动态共享内存** | 启动传的是 `shared_mem_bytes = 0`；只用到静态 `.shared`，而 `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` 从不被调用，因此一个需要超过每 block 默认上限的内核会在 JIT 时失败。设备工厂会预先拒绝一台上限低于 profile `shared_max`（48 KiB）的设备。 | `device/src/cuda/program.rs`、`runtime/src/devices/cuda.rs` |
| **Hopper / Blackwell 矩阵路径** | 只降低了 `mma.sync`（`m16n8kK`）；没有 `wgmma`，没有 `tcgen05`。 | `codegen/src/llvm/nvptx/wmma.rs` |
| **预汇编的对象** | 对象缓存存的是 PTX 文本；每一次新鲜加载都要付出驱动 JIT 的代价（驱动会把它缓存在 `~/.nv/ComputeCache`）。`ptxas` 预汇编（`object_format: cubin`）尚未接线。 | `runtime/src/devices/cuda.rs` |
| **用户态 NV 驱动** | Tinygrad 的 `ops_nv`（直接的 GPU-FIFO 提交）需要为每个驱动分支生成一套 ABI；Svod 留在稳定的 `libcuda.so.1` API 上。`NV` 在 `SVOD_DEVICE` 中被接受为 `CUDA` 的别名，并为那个未来的后端保留。 | `nvidia_backend_plan.md` |

算是数值上的注记而非缺口：f64 的 `Exp2` / `Log2` 以及全部超越函数都走多项式
路径（[代码生成](./codegen.md)）；`lg2.approx.f32` 对渲染器可用，但普通的图
不会用到它。

---

## 今天没得商量的要求

- 驱动至少要到 CUDA 12.0 / R525：CUDA graph 的入口点按其 12.0 的版本化名称绑定。
  PTX ISA 由 `--cuda-feature=+ptx78` 锁定在 7.8，因此更新的 clang 不会抬高这个下限。
- `clang` 必须带有 NVPTX target；没有 NVRTC 回退。

---

## 路线图

按计划（`nvidia_backend_plan.md`，第 5 阶段）列出的顺序：

1. **带作用域的同步**：为每个缓冲区建一张生产者表，让宿主访问等待 event
   而不是 `cuCtxSynchronize`。
2. **真正的 P2P**：绑定 `cuDeviceCanAccessPeer` / `cuCtxEnablePeerAccess` /
   `cuMemcpyPeerAsync`，并把 `SyncStrategy::PeerToPeer` 路由到它们上面。
3. **CUPTI 计数器**：把 `PmcCounter` 拓宽到 AMD SQ 那一组之外，并加上一个
   第 4 层提供者。
4. **fp8**：降低 sm_89 的 `cvt` intrinsic，好让 fp8 的 `mma.sync` 行变得可达。
5. 当 toolkit 存在时做 **`ptxas` 预汇编**，并作为 cubin 缓存。
6. 经由 `cuFuncSetAttribute` 支持**动态共享内存**，以及让 `tk` 走 `GpuArch`，
   使 tile 内核能在 CUDA 上运行。
