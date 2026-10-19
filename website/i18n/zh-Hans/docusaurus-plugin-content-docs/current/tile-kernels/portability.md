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

| 目标 | 原子 | 填充 | 配置表 | 状态 |
|---|---|---|---|---|
| CUDA sm_80+ (sm_86) | `mma.sync` m16n8k16, `ldmatrix` | `cp.async`，2–3 级 | 有 | 在 RTX 3060 上测量 |
| CUDA sm_90 (Hopper) | `mma.sync` m16n8k16, `ldmatrix` | `cp.async`，共享内存最多 227 KB | sm_80 配置表 | 已编译：每个族都能用 `ptxas -arch=sm_90` 和 `sm_90a` 汇编；未运行 |
| AMD RDNA4 (gfx1200, gfx1201) | WMMA 16×16×16，每 lane 8 个值 | 寄存器中转，2 级 | 有，未测量 | 已编译为 code object；未运行 |
| AMD RDNA3 / RDNA3.5 (gfx1100–1102, gfx1151) | WMMA 16×16×16，输入复制 | 寄存器中转，2 级 | 有，未测量 | 已编译为 code object；未运行 |
| AMD CDNA3 / CDNA4 (gfx942, gfx950) | MFMA 16×16×16，wave64 | 寄存器中转，2 级 | 有，未测量 | 已为 gfx942 编译；未运行 |
| Apple | simdgroup 8×8×8 | 无 | 无 | 仅有原子 |

`ops::supported(device)` 仅在有配置表的地方为真；在其他设备上，每个 op 构建自己的图回退（`Fallback::Target`）。主机测试把每个原子的操作数布局与厂商 ISA 文档中的 lane 公式逐一核对；对每个有配置表的目标，它们降级每个内核族，在降级前后用解释器运行，并在安装了 `ptxas` 或 clang 时编译。"已编译"不等于"在设备上正确"：只有当 `targets::families_match_the_interpreter_on_the_device` 在该硬件上通过时，目标才算已测量。

没有 `cp.async` 时（RDNA 没有 global → LDS 拷贝），流水线把每一步加载到寄存器，计算上一步，再把寄存器写入共享内存，使用两个槽位。同一路径也在 CUDA 的设备测试中运行（`register_staged_families_match_on_cuda`）。

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
| Hopper (sm_90a)：wgmma、TMA、mbarrier、warp 专用化模板 | 未开始；Hopper 走 `mma.sync` 路径（基础设施第 5–8 项） |
| Blackwell (sm_100a)：tcgen05、张量内存 | 未开始；没有 B200 可用 |
| AMD CDNA：ping-pong 模板、MFMA 32×32、`buffer_load … lds` | 未开始 |
| AMD 测量 | 尚无 AMD 内核运行过；配置表只是供 tune store 挑选的初始猜测 |
| AMD 注意力寄存器 | 分数 tile 经共享内存进入 P·V，V 操作数逐元素收集；RDNA 上 d = 128 会溢出寄存器 |
| Apple | 仅有原子；没有配置表，没有发射的内核 |
| Warp 角色、角色屏障、原始 asm 语句 | 可在 IR 中记录，但被降级拒绝 |
| 卷积（隐式 GEMM） | 在 sm_86 上测量，YOLO26 以 channels-last 运行在其上（`ops::conv2d`）；对 AMD 仅编译 |
| fp8/int8 权重、GEMV + argmax、持久化网格、注意力反向 | 未开始 |
