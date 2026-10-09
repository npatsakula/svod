---
sidebar_label: 可移植性
---

# 可移植性

## 目标 {#targets}

`atoms::Target` 是降级所了解的关于 GPU 的全部信息：

| 字段 | 含义 |
|---|---|
| `arch` | `GpuArch::{Cuda, Amd, Metal}` |
| `wave` | 每个 warp 或 wave 的 lane 数 |
| `mma` | 带操作数布局的矩阵核心原子 |
| `cp_async` | 异步的全局 → 共享拷贝（CUDA sm_80+） |
| `ldmatrix` | warp 协作的 8×8 b16 片段加载（CUDA sm_75+） |
| `smem_bytes` | 每个块的共享内存（设备报告时为 opt-in 上限） |
| `sms` | SM 或 CU 数量（设备报告时） |

`Target::for_device(&spec)` 从实际设备解析目标。`Target::for_arch(arch)` 仅根据架构构造目标，主机测试使用的就是它。`atoms::sm86()` 是带 28 个 SM 的 RTX 3060 目标。

| 目标 | 原子与布局 | 配置表 | 已降级并运行 |
|---|---|---|---|
| CUDA sm_80+（在 sm_86 上测量） | `mma.sync` m16n8k16, `ldmatrix`, `cp.async` | 有 | 是 |
| AMD CDNA (gfx942) | MFMA 16×16×16 | 无 | 否 |
| AMD RDNA3 / RDNA4 | WMMA 16×16×16 | 无 | 否 |
| Apple | simdgroup 8×8×8 | 无 | 否 |

`ops::supported(device)` 只在存在配置表的地方为 true，因此在其他所有设备上，每个算子都构建其计算图回退（`Fallback::Target`）。AMD 和 Apple 的原子已存在，主机测试也检查了它们的布局能铺满指令形状，但还没有任何内核在这些目标上被降级或运行。

## 策略 {#strategy}

已确定的做法是**每个算子一个共享的 tile 程序**。像 `kernels/gemm.rs` 这样的内核基于 tile 值只写一次，从不按厂商分支。按目标变化的是由降级和算子层选择的数据：

| 按目标变化 | 位置 |
|---|---|
| 矩阵核心原子及其布局 | `atoms`, `layout/atoms.rs` |
| 拷贝机制（`cp.async`、`ldmatrix`、寄存器分级） | `Target` 标志、发射器 |
| 调度模板 | `schedule::Schedule` |
| 配置候选表 | `ops::config` |

为某个厂商分叉出单独的内核程序只允许作为有限的例外。

## 尚未完成 {#not-done}

| 项目 | 状态 |
|---|---|
| Hopper (sm_90a)：wgmma、TMA、mbarrier、warp 专用化模板 | 未开始；将在远程 H100 硬件上测量 |
| Blackwell (sm_100a)：tcgen05、张量内存 | 未开始；没有 B200 可用 |
| AMD CDNA：ping-pong 模板、MFMA 32×32、`buffer_load … lds` | 未开始；将在远程 MI300X 硬件上测量 |
| AMD RDNA、Apple | 仅有原子；没有配置表，没有发射的内核 |
| Warp 角色、角色屏障、原始 asm 语句 | 可在 IR 中记录，但被降级拒绝 |
| 卷积（隐式 GEMM） | 仅 CUDA sm_80+（`ops::conv2d`）；在有配置表之前 AMD 仍走图或 tk1，YOLO 模型尚未接入 |
| fp8/int8 权重、GEMV + argmax、持久化网格、注意力反向 | 未开始 |
