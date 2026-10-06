---
sidebar_label: AM 驱动
---

# AM 驱动（用户态）

**AM** 是位于 `device/src/amd/am/` 之下、实验性的、尚不可选择的用户态驱动，它直接
与 AMD GPU 的 PCI BAR 对话，而不经过 `amdgpu`/KFD。它在思路上追随 tinygrad 的 AM
驱动，但面向的是另一类硬件（虚拟功能而非裸机）。它只是**脚手架**：纯逻辑部分已
实现并有单元测试，初始化流程曾在一块真实 GPU 上运行过一次，但从未有任何内核通过
它执行。

:::caution[不可选择，不可运行]
`SVOD_AMD_BACKEND=am` 会被拒绝（`unknown SVOD_AMD_BACKEND=am (only 'kfd'
supported)`，`device/src/amd/device.rs`）：`am/` 中没有任何部分实现
[`AmdIface`](./overview.md) 接缝，因此 `AmdDevice` 无法使用它。这些代码只能通过
独立的 `device/examples/am_*.rs` 程序访问。它最后一次功能性改动是在 2026 年 6 月；
之后的提交都是清理。下面的状态描述的是那次提交实际运行到的内容，而不是持续性的
保证——它没有 CI 或硬件测试。
:::

该模块在每台 Unix 宿主上编译（`cfg(unix)`，与后端其余部分一样），因此它总会经过
类型检查和 lint，其逻辑也有单元测试（`device/src/test/unit/amd/am/` 下约 40 个
测试：页表、TLSF、内存管理器、寄存器表、discovery 解析器）。

---

## 目标：CDNA3 SR-IOV VF（gfx9.4.x）

`AmDev::open` 只接受 GC IP 版本为 9.4（CDNA3）的 GPU 的 **SR-IOV 虚拟功能**——
初始化所用硬件是一个直通进 KVM 客户机的 MI300 级 VF。非 VF 的功能或其他 GC 版本会被
拒绝（`device/src/amd/am/dev.rs`）。gfx11 页表编码仍然保留实现并有单元测试，gfx9
路径复用的正是它的几何结构与物理范围辅助函数。

作为 VF 而非裸机，这一点塑造了整个驱动：

- **GC MMIO 由宿主把关。** 直接读取 GC 寄存器会返回 `0xffffffff`；GC / GCVM
  寄存器要**经由 RLC 间接**访问（把值暂存到 RLC scratch，触发 `RLC_SPARE_INT`，
  然后轮询）。
- **VRAM 和 IP discovery 在获得授权前不可访问。** 宿主的 **GIM**（SR-IOV 宿主
  驱动）必须通过一次 **mailbox 握手** 授予访问权限，该握手在 discovery 之前运行。
- **宿主 PF 拥有特权子系统：** PSP、SMU、时钟、固件 / world-switch，以及
  **doorbell aperture 路由**。AM 编程每个 VF 的状态（页表 context0、每个引擎的
  失效范围、TLB 刷新、环/队列 MQD），并尽力写入少数归 PF 所有的寄存器（L2 缓存
  配置、系统与身份 aperture、`GB_ADDR_CONFIG`、`RLC_CNTL`、`SH_MEM_BASES`），
  忽略被拒绝的写入。

tinygrad 的 AM 正好相反：只支持裸机，解绑 `amdgpu` 并独占整个设备。VF 变体需要
mailbox、RLCG 间接路径以及仅限单个 VF 的 hub 编程，并且永远无法独占各个引擎。

---

## 已有内容

| 分组 | 模块 | 作用 | 状态 |
|---|---|---|---|
| Discovery | `pci.rs`、`discovery.rs` | sysfs BAR mmap（BAR0 VRAM / BAR2 doorbell / BAR5 MMIO）、配置空间读写、带边界检查的 IP-discovery 解析器（每个 XCC 的段基址，`gc_info` v1/v2） | 已在 VF 上运行；解析器有单元测试 |
| 寄存器访问 | `regaccess.rs`、`rlcg.rs`、`mailbox.rs`、`regs.rs`、`regs_gen.rs` | VF↔GIM mailbox 握手、按 XCC 的 RLCG 间接 GC/GCVM 读写、MMIO/RLCG 路由器、带 `select` / `find` / `encode` 的 vendored 寄存器表 | 已在 VF 上运行（scratch 回显、每个 XCC 上的 `GRBM_STATUS`）；表逻辑有单元测试 |
| 内存（GMMU） | `mm/{tlsf,pagetable,manager,mod}.rs` | 用于 VA、物理 VRAM 和页表池的 TLSF 分配器；4 级 / 48 位遍历（`va_shifts = [12, 21, 30, 39]`）；gfx9 和 gfx11 的 PTE/PDE 编码；大页；表回收；`valloc` / `vfree` | 有单元测试；页表经 BAR0 写入 VRAM 并由 CPU 回读遍历——尚未确认有 GPU 通过它们进行地址转换 |
| GMC 初始化 | `ip/gmc.rs` | 两个 hub 的 context0（base/start/end + CNTL）、MX_L1_TLB、每个引擎的失效范围、ENG17 TLB 刷新、HDP 刷新、原始故障状态读取 | 已在 VF 上运行到上下文编程，每个 XCC 上都收到刷新 ACK |
| GFX 初始化 | `ip/gfx.rs` | MEC 启用（不检查的写入）、v9 计算 MQD、HQD 激活、`WRITE_DATA` PM4 | `CP_HQD_ACTIVE` 读到 1；队列从未消费过数据包 |
| SDMA 初始化 | `ip/sdma.rs` | 解除 F32 停机，编程 RB base/rptr/wptr + doorbell，提交，`wait_idle` | 已编程；拷贝从未完成 |
| 编排器 | `dev.rs` | `AmDev::open` = mailbox → discovery → GMMU → GMC context0 → 刷新；`valloc`、`vram_read` / `vram_write`、`release` | 已在 VF 上运行至 GMC |

页表由一个可注入的 `PhysMem` trait 提供后备存储——测试中是普通缓冲区，驱动中是
BAR 映射的 VRAM（`VramPhys`）。叶子编码是唯一与架构相关的部分：gfx9 把 MTYPE 放在
第 57 位，在 PDB1 表项上设置 `bfs`、在 PDB0 表项上设置 translate-further，并把
PDB1/PDB2 叶子标记为 `PDE_PTE`；gfx12 为 `unimplemented!`（常量已记录；有一个测试
断言该 panic）。

### 寄存器表只生成一次，然后 vendored

tinygrad 是一个有时不存在的子模块，因此构建从不依赖它。
`device/tools/gen_am_regs.py` 需手动运行：它解析 tinygrad 的
`autogen/am/regs.py`，并输出已提交的 `am/regs_gen.rs`。启动时 `select` 选择主版本
相同且 `≤ ip_ver` 的最大表版本。已提交的表涵盖 gfx9.4.3 集合（`gc_9_4_3`、
`mmhub_1_8_0`、`osssys_4_4_2`、`sdma_4_4_2`、`nbio_7_9_0`、`hdp_4_4_2`、
`mp_11_0_0`、`mp_13_0_0`）和 gfx11.5.0 集合（`gc_11_5_0`、`mmhub_3_3_0`、
`mp_14_0_2`、`nbio_7_11_0`、`hdp_6_0_0`、`osssys_6_0_0`）；KFD 路径的硬件计数器
（`amd/pmc.rs`）用的也是 gfx11 GC 表。

---

## 示例

每个 `device/examples/am_*.rs` 程序都是一个独立的初始化验证工具。2026 年 6 月那次
运行确认了以下结果：

| 示例 | 作用 | 结果 |
|---|---|---|
| `am_discovery` | BAR 映射 + IP discovery，只读；可与已绑定的 `amdgpu` 共存 | 枚举出 8 个 GC 9.4.3 实例、SDMA 和 AID |
| `am_own` | mailbox 授权 + RLCG scratch 回显 + 每个 XCC 上的 `GRBM_STATUS` | 断言通过 |
| `am_gmc` | 编程 GC + MM context0；每个 XCC 上的 ENG17 TLB 刷新 ACK；打印故障状态 | 每个 XCC 都收到 ACK |
| `am_sdma` | SDMA 环设置 + 经由它的一次拷贝 | 引擎不消费该环 |
| `am_compute` | MEC 启用 + MQD 激活 + `WRITE_DATA`，同时通过 doorbell 和直接写 `CP_HQD_PQ_WPTR` 触发 | HQD 被激活；哨兵值始终没有写入 |

障碍在于引擎交接：doorbell aperture 路由和引擎启动归宿主 PF 所有。从 VF 启用该
aperture（`_PF` BIF doorbell 寄存器）会卡死 VF↔GIM mailbox，需要重启虚拟机，因此
`enable_doorbell_aperture` 虽存在于 `ip/gfx.rs` 中，但在 VF 上被标记为禁止调用，
并在 `am_compute` 中被注释掉。

---

## 尚不存在的内容

- **`AmdIface` 实现**——因此 AM 不能作为设备后端。
- **PSP 固件加载**、**SMU / 时钟**——在 VF 上归 GIM 所有；在裸机上它们将是规模最大、
  风险最高的移植部分。
- **中断处理程序**——没有 `ip/ih.rs`；OSSSYS 表只被 `am_discovery` 使用。初始化
  采用轮询。
- **GPU 引擎通过 AM 页表执行工作的证据**。

仅为 GMC 初始化存在两个调试开关：`SVOD_AM_DEBUG`（任意值）在写入寄存器后回读并
记录被拒绝的 GC 写入，`SVOD_AM_MCBASE`（`raw`、`fb` 或 `fbxgmi`）覆盖 MC
aperture 基址。

---

## 为什么保留它

动机来自内核调度器：在单 XCC 的 gfx11+ 部件上，激进的多队列派发可能让 CP 微引擎
停在 MES 固件无法抢占的等待中，这正是 [通道池](./queues-and-dispatch.md) 保持保守
的原因。独占 GPU 就能把内核从派发路径中移除。这一论点目前还不适用于 AM 所支持的
场景——在 VF 上，调度、world-switch 和 doorbell 仍归宿主所有——而要在裸机上做到，
则需要在现有内容之上再加上 PSP、SMU、中断以及接缝实现。如今的设计价值在于接缝本身：
如果某天 AM 后端真正落地，它实现的是与 `KfdIface` 相同的五个方法和三个钩子，接缝
之上的一切都无需改变。
