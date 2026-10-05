---
sidebar_label: 调试
---

# 调试与故障分诊

当一个 GPU 内核触碰了它不该碰的内存时，KFD 报告一个故障，只带着一个原始的
虚拟地址，几乎别无其他。本页介绍后端把这个地址变成诊断所拥有的
工具：对故障地址进行分类的 VA→分配注册表、干净地停下设备的
poison 闩锁，以及调度/追踪插桩。

---

## 问题：一个光秃秃的故障 VA

一个 `WAIT_EVENTS` 内存故障事件交回一个 `kfd_hsa_memory_exception_data`，
带有故障的 `va`、失败标志（`NotPresent`、`ReadOnly`、`NoExecute`、
`imprecise`），以及一个 `ErrorType`。那告诉你 GPU 在*哪里*故障，却没告诉你
*那里有什么*——而后者才是真正定位 bug 的问题。最终把它
浮现出来的那次 panic 是在下一次 `synchronize()` 时的延迟重抛，
远离故障现场。

---

## VA 注册表

`device/src/amd/va_registry.rs` 是一张诊断旁表，它把每个活跃的
GPU VA 范围映射回其所属的分配。它是纯记账——没有 GPU
依赖——因此分类逻辑可在任何宿主上进行单元与属性测试。一个 `VaRegistry`
存在于 `KfdIface` 上（一次故障会损坏整个 VM，所以每设备是
正确的粒度）。

它在一个分配生命的两端被维护：

- **`alloc_raw`** 在 `MAP_MEMORY_TO_GPU` 成功之后调用
  `va.insert(base, size, handle, tag)`。
- **`free_raw`** 在 unmap *之前*调用 `va.remove(base)`——这样一个落在
  刚被释放的 VA 上的故障会被分类为 use-after-free，而不是一个活跃的
  分配。

### 标签

每个分配都被打上其用途的标签（`AllocTag`）。`Vram` 与 `Gtt` 是
由 `AllocKind` 推导出的默认值；更细的标签则由 `alloc_*_tagged`
的各个调用点显式传入：

| 标签 | 涵盖 |
|---|---|
| `Vram` | 通用设备 VRAM——张量数据、code object、EOP/ctx-save |
| `Gtt` | 宿主可见的控制内存——GTT，以及图 / 已链接 plan 的 IB 与控制缓冲区 |
| `Kernarg` | kernarg arena（宿主可见的 VRAM）——每调度、图以及已链接 plan 的参数页 |
| `SignalPool` | GTT 信号槽池 |
| `QueueRing` / `QueueGart` / `QueueInactive` | 一个队列的环、GART 页与 queue-inactive 信号 |
| `Staging` | GTT 上的 SDMA 反弹缓冲区 |
| `Scratch` | 寄存器溢出的 scratch——仅 GPU 的 VRAM，每内核重分配 |

真正要紧的区别是 **scratch 与其他一切**：scratch 是
唯一一个共享、仅 GPU、被动态重分配并释放的区域，也是历史上的
`NotPresent` 元凶。每进程的 KFD 事件页在 `alloc_raw` 之外分配，
不被追踪：发生在那里的故障会被归类为未映射。

### 分类

注册表保有一个活跃范围的 `BTreeMap`（以基 VA 为键，便于范围
查询），外加一个有界的、最近释放的 **256** 个区域的环
（`FREED_HISTORY`）。`classify(va)` 以这一优先级解析一个故障地址：

```text
1. Live    — va is inside a currently-mapped allocation
             (live takes precedence, so a re-allocated VA reads Live, not stale)
2. Freed   — va is inside a recently-freed region → use-after-free
3. Unmapped — va is in no tracked region; report nearest live neighbours + gaps
```

落进故障消息的是 `Display` 的渲染：

```text
Live:     va is at offset +0x40 within a LIVE scratch allocation
          [0x7f…000, 0x7f…400) (handle=0x42)

Freed:    va is within a RECENTLY-FREED scratch region [0x…, 0x…) (handle=0x…)
          — use-after-free: a stale/recycled VA still referenced by an
          in-flight kernel

Unmapped: va is in NO tracked allocation; nearest live below: VRAM buffer
          [0x…, 0x…) (va is +0x80 past its end); nearest live above: …
```

---

## 一次故障是如何被报告的

在 `KfdIface::wait_events`（`device/src/amd/iface.rs`）中，当内存故障
事件已触发（`gpu_id != 0`），字段会从 bindgen 生成的 union 载荷中被
复制进局部变量，VA 被分类，并构建出一条丰富化的消息：

```text
AMD GPU memory fault on gpu_id=… va=0x… (NotPresent=true ReadOnly=false NoExecute=false
Imprecise=false ErrorType=…) — va is at offset +0x40 within a LIVE scratch …
```

它通过一个 `fault_logged: AtomicBool` 闩锁和一个
`tracing::error!` **只记录一次**。这一点很要紧：内存故障事件不会
自动重置，因此后续的 poll-fault 调用（`wait_events(0)`）会重新观测到同一个
故障——每次都记录会刷屏。它随后作为一个有类型的
`Error::GpuFault` 返回，其 `Display` 就是上面那个字符串；poison 闩锁
则会记录该文本，并在此后的每一个入口点把它重抛为 `Error::Runtime`（带有
`runtime error: ` 前缀）。一次未遇故障而超时的等待则是另一个不同的错误
`Error::TimelineTimeout`。
（一个硬件异常事件，槽 `[2]`，改为报告
`reset_type`/`reset_cause`/`memory_lost`——这些没有可分类的故障 VA。）

---

## poison 闩锁

一次内存故障会损坏整个每 VM 的页表，因此设备在
一次故障后就死了。`AmdDeviceCore`（`device/src/amd/device.rs`）持有一个 poison 闩锁——
`poisoned: AtomicBool` + `error_msg: OnceLock<String>`——在每个
调度与 synchronize 入口点处检查：

- `poison(msg)` 把消息记录一次并置位标志；
- `is_poisoned()` 是热路径上的门；
- `poison_error()` 在被毒化时返回已记录的 `Error::Runtime`；
- `poll_faults_nonblocking()` 在一次信号等待到达其截止时间时（调度、复制与
  通道获取均为 30 秒）发出 `wait_events(0)`，这样附到超时上的就是真实错误，
  而不是一个光秃秃的截止时间。（自旋升级路径同样会在故障时提前跳出，
  但走的是一次短暂的*阻塞式* `wait_events`，而非这个 poll。）`poison()` 还会
  唤醒每一个停驻在通道池上的线程。

一旦被毒化，对该设备上任何通道的每一次 `synchronize`/`execute`
都会快速失败——GPU 状态和缓存的映射不再可信。

---

## 调度插桩：`SVOD_DEBUG_DISPATCH`

设置 `SVOD_DEBUG_DISPATCH`（设成任何值）会在两个点打开 `eprintln`
转储，二者都位于 `device/src/amd/program.rs`：

- **`[program-load]`**——每程序：kernarg/private/group 尺寸、
  `kernel_code_properties`（逐位解码）、user-SGPR 计数、`wave32`，
  以及原始的 `rsrc1/2/3`。它会标出加载器*没有*填充的
  `kernel_code_properties` 位（那会让内核读到垃圾指针并故障）。
- **`[dispatch tv=…]`**——每调度：内核名、`grid`、`local`、`is_pm4`、
  kernarg 的 GPU VA、scratch VA，以及每个缓冲区的 VA。

这是看清一次故障调度究竟触碰了哪些 VA 的最快方式，以便
与注册表的分类相互参照。

---

## 追踪设置（`RUST_LOG`）

后端使用 `tracing` crate（`debug!`、`tracing::error!`），但
**不安装任何 subscriber**——那是宿主二进制的职责。`alloc_raw`/`free_raw` 的
`debug!` 行以及那条一次性的故障 `error!`，只有在安装了 subscriber
且级别允许时才会出现。

那些会安装 subscriber 的示例二进制在 `main` 中调用
`tracing_subscriber::fmt::init()`（它遵循 `RUST_LOG`）：

```bash
# Surface the alloc/free debug lines and the fault error from gigaam_infer:
RUST_LOG=svod_device=debug \
SVOD_DEVICE=AMD:0 \
  cargo run --release -p svod-model --example gigaam_infer -- ./audio.wav
```

:::tip[流水线调试器]
对于*编译器*侧的问题（IR 提取、LLVM IR、UOp 树）而非
驱动，项目附带一个 `/svod-debug` skill，记录了前端 →
codegen 的追踪目标（`SVOD_DUMP_LLVM_IR`、`SVOD_DUMP_AMD_IR`、每阶段的
`RUST_LOG` 目标、`setup_test_tracing()`）。那与本页面向驱动侧的
故障分诊是一套独立的工具箱。
:::

---

## 一次实战分诊

当一次 `NotPresent` 故障复现时，工作流是：

1. 故障消息已经点名了类别——先读它。"LIVE scratch"
   指向 scratch 重分配路径；"RECENTLY-FREED"是一个
   在内核仍引用时就被释放的缓冲区的 use-after-free；"NO tracked allocation"
   且附近有活跃邻居则是一次越界（那个 gap 告诉你越了多少）。
2. 设上 `SVOD_DEBUG_DISPATCH` 重跑，以看清那次故障调度的确切 VA，
   并设 `RUST_LOG=svod_device=debug` 以看到直到那一刻的 alloc/free 历史。
3. 把故障 VA 与转储出来的 scratch/kernarg/缓冲区 VA 相互参照。

`NotPresent` 的头号嫌犯是 **scratch**（按 `Scratch` 标签）——
唯一一个共享、仅 GPU、被动态重分配并释放的区域，在那里一次
重分配与调度的竞态会让一个内核指向一个已释放的缓冲区。

---

## 为什么这很重要

在有注册表之前，一次故障只给你一个十六进制地址，别无其他。如今
故障*消息本身*就说明该 VA 是活跃 scratch、一个已释放/陈旧的 VA，还是
野指针——把一次盲目的追猎变成一次有方向的。配上 poison 闩锁
（它干净地停下设备，而不是让损坏的状态扩散）
和调度转储，后端无需给 GPU 挂调试器就能定位一次内存
故障。
