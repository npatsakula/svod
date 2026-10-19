---
sidebar_label: 可移植性
---

# 可移植性 {#portability}

## 目标 {#targets}

`atoms::Target` 记录 GPU 能做什么：它的矩阵核心原子、全局内存如何到达共享环形缓冲（`Fill::CpAsync` 或 `Fill::Staged`）、`ldmatrix` 与转置加载、lane 移动、哪些同步是指令（`SyncCaps`）、占用率、共享内存和 SM 数量；详见其 rustdoc。`Target::for_device(&spec)` 从实际设备解析目标，`Target::for_arch` 仅根据架构构造目标（主机测试使用的就是它），而 `Target::offline(arch, sms, xcds)` 或 `Target::card(name)` 描述本机没有对应设备的显卡。

目标只声明芯片能做什么，从不声明其代价。其原语的代价（`atoms::Costs`）在首次使用时于设备上测量（`probe::costs`），并保存在调优存储中；主机测试则直接声明它们（[调优](./tuning#device-costs)）。不存在按家族划分的代价表。这些代价以 `Lowering::device` 进入降级（方向、交换或往返），以 `Planner` 的设备进入算子层（候选排序）。已测量的目标及其数字见[概览](./overview#targets)。

当 `device` 是默认设备、其目标有 tk 内核（`config::has_kernels`）并报告了 SM 数量时，`ops::supported(device)` 为真；代价探针在第一次询问时运行。在其他任何设备上，每个算子都构建自己的计算图回退（`Fallback::Target`）。主机测试把每个原子的操作数布局与厂商 ISA 文档中的 lane 公式逐一核对；对每个有内核的目标，它们降级每个内核族，在降级前后用解释器运行，并在安装了 `ptxas` 或 clang 时编译。“已编译”不等于“在设备上正确”：只有当 `targets::families_match_the_interpreter_on_the_device` 在其硬件上通过后，目标才算已测量。

没有 `cp.async` 时（AMD 没有异步的全局 → LDS 拷贝），流水线把每一步加载到寄存器，计算上一步，再把寄存器写入共享内存，使用一个或两个槽位。同一路径也在 CUDA 的设备测试中运行（`register_staged_families_match_on_cuda`）。

## 策略 {#strategy}

已确定的做法是**每个算子一个共享的 tile 程序**。像 `kernels/gemm.rs` 这样的内核基于 tile 值只写一次，从不按厂商分支。按目标变化的是由降级和算子层选择的数据：

| 按目标变化 | 位置 |
|---|---|
| 矩阵核心原子及其布局 | `atoms`, `layout/atoms.rs` |
| 拷贝机制（`cp.async`、`ldmatrix`、寄存器暂存、转置加载） | `Target` 字段、发射器 |
| 调度模板 | `schedule::Schedule` |
| 配置候选排序 | `ops::config`，依据测得的 `Costs` |

为某个厂商分叉出单独的内核程序只允许作为有限的例外。

## 本版本未包含 {#not-in-this-release}

| 项目 | 状态 |
|---|---|
| Hopper TMA、mbarrier、warp 专用化生产者 | 未开始；GEMM 和注意力运行由 `cp.async` 供数的 `wgmma` |
| Blackwell `tcgen05`、张量内存 | 未开始；sm_120 走 sm_80 路径 |
| CDNA MFMA 32×32、`buffer_load … lds` | 未开始（32×32 布局已存在，但没有原子） |
| CDNA 之外的双工作组乒乓 | 只有在矩阵核心与向量单元并行运行、且一个 lane 的寄存器文件容纳得下两个工作组的累加器的地方（CDNA），`Target::crews` 才为 2 |
| Apple | 仅有原子；没有内核 |
| 卷积的图像暂存形式、深度可分离卷积、分组卷积 | 计算图 |
| fp8/int8、注意力反向传播 | 未开始 |
