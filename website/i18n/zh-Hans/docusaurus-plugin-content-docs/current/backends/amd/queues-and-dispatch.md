---
sidebar_label: 队列与调度
---

# 队列与调度

AMD 后端保留了 Tinygrad 已验证的 PM4、AQL 与 SDMA 数据包语义，
但在队列调度与故障处理上改用 Rust 的所有权。核心规则很简单：
**一份不可克隆的租约是一条计算通道唯一的发布权威**。

## 计算通道

`AmdDeviceCore` 拥有一个有界的 `QueuePool`。它的槽位是固定的 `OnceLock`，
队列则按 `SVOD_AMD_HW_QUEUES` 惰性创建，该值被夹在 1 到 64 之间，
在多 XCC 的 CDNA 上默认为 4，其余一切默认为 1。一个原子位集追踪租约：

- 认领一条已初始化的空闲通道是一次原子 compare-exchange；
- 队列创建是一条冷的串行化路径；
- 当每条通道都已被租出时，调用方在一个条件变量上停驻；
- 丢弃 `QueueLease` 会清掉相应位并唤醒一个等待者；
- 队列绝不与宿主发布方共租。

程序与图模板从不持有 `QueueLease`。`OwnerCtx` 装的是
逻辑 plan 状态——设备 core 与分配器、最新的完成、PMC 选择、
一个可选的已链接重放模板——以及一个 `session` 槽位，
只有直接调度回退会用它的 epoch 租约填充该槽位。

一个线程每次等待最多停驻 30 秒；连续两次超时会针对「AMD lane acquisition」
返回 `Error::TimelineTimeout`，而一个被毒化的设备会以其毒化错误唤醒每一个等待者。

直接语义回退在一个重放 epoch 中的所有内核之间保持同一份租约，
随后由 `PlanContext::finish_replay` 释放它。后一个 epoch 会先等待上一个
finalizer 才去获取另一条通道，因为换一个队列并不会继承旧队列的 FIFO 顺序。
图与原生的已链接重放本就会在复用其可变的 kernarg/控制存储之前等待，
并为每一个发布 epoch 租用一条通道。

## 原生队列

`AmdComputeQueue` 拥有一个 16 MiB 的宿主可见环、GART 读/写指针、
一份 doorbell 映射，以及 KFD 队列后备。数据包格式只选定一次：

```text
PM4 = num_xcc == 1 && SVOD_AMD_AQL is unset or "0"
AQL = otherwise
```

- PM4 队列发布原始 dword，并敲响下一个 dword 索引。
- AQL 队列发布 64 字节数据包，并敲响最后写入的那个数据包的索引。
- AQL 内核的 `completion_signal` 保持为零。厂商 IB 的 PM4 wait/store
  自行负责 timeline 完成，在多 XCC 硬件上配以 XCC0 的 `PRED_EXEC`。

通道租约消除了计算侧的共租。`AmdComputeQueue.inner` 仍使用一个互斥锁
作为 Rust 别名保护；在正常的计算路径上它是无竞争的。
单例的 SDMA 队列被独立地以互斥锁保护，因为来自不同 plan 的复制可能共享它。

## 发布

原生的已链接重放把一次提交拆分为准备与发布两步：

1. 校验程序身份、具体缓冲区的所有权、ABI、启动几何、
   补丁表以及硬件流限制。
2. 预留并写入 kernarg/控制数据。
3. 获取环的余量。
4. 当设备级排空需要观测一个 plan 自有的 timeline 时，注册一个已准备的
   finalizer。
5. 发布数据包与 doorbell。
6. 把该 finalizer 标记为已发布。

如果注册之后有错误向上展开，那个已准备的 finalizer 会变为失败状态。
一次并发的排空会被唤醒并立即失败，而不是去等待一个从未发布过的
终结存储。物理设备随后被毒化，于是该通道无法再被复用，
被硬件引用的分配则被隔离。

普通的逐内核调度（`submit_hcq_dispatch`）更简单：它的 finalizer
在 doorbell 之后即以已发布状态诞生，由调用方将其登记为在飞；doorbell
之前的失败则会回滚环与 timeline 预留。

PM4、AQL 与 SDMA 的发布在环回绕之前都会检查 KFD 读指针（AQL 与 SDMA
读取的是一个单调值；PM4 指针是相对于队列的，其 epoch 由生产者索引重建）。
普通调度还会额外为在飞的 timeline 值设界。PM4 的 timeline 值会在 2^31
水位处排空并复位，因为硬件的 wait/store 数据包比较的是低 32 位。

## 资源生命期

每一个直接提交的 finalizer 都持有其 code object。图与已链接的 plan
持有它们所链接的全部 code object。持久化的 kernarg、常驻命令、控制、
时间戳与 PMC 分配都保持被拥有，直到其确切的重放完成被退休为止。

队列的生命周期是显式的：

```text
Constructing -> Active
Constructing -> Destroyed | Quarantined
Active -> Destroyed
Active -> Quarantined
```

有序的计算侧拆除依次是排空、KFD `DESTROY_QUEUE`、scratch 释放，
然后是环/GART/上下文释放。一次失败的排空或销毁会毒化物理设备，
并让所有可能被引用的后备保持映射。队列销毁成功之后的 doorbell
取消映射失败会被报告为一次宿主映射泄漏，但不会不必要地隔离
安全的 GPU 后备。

如果 `CREATE_QUEUE` 成功，而 doorbell 映射与回滚销毁双双失败，
`setup_ring` 会返回 `AmdQueueStillActive`。调用方会在分配守卫展开之前
毒化设备，以免一个活着的 KFD 队列观测到已被释放的环内存。

一次穿过某条通道向上展开的 panic 会隔离该通道，但不会毒化设备；只有在其
doorbell 敲响*之后*被遗弃的预留（`TimelineReservation`、`PreparedPublication`）
才会毒化设备，无论起因是错误还是 panic。信号槽会无条件地归还给池，除非设备
已被毒化——此时它们会被扣留，因为一个被遗弃的队列可能仍在瞄准它们。

## 设备级排空

每条通道拥有一条队列 timeline 和一个非队列 finalizer 的 FIFO。设备 core
对每一条已初始化的通道保有弱引用。`synchronize_all` 会为这些通道拍下
快照，并在不取发布锁的情况下等待它们的 timeline。宿主的读、写以及
破坏性的释放优先使用作用域化的 `wait_storage`，它只等待针对那个存储基址
记录在案的提交，而在 VA 未知时或在 `SVOD_AMD_SCOPED_SYNC=0` 之下
回退到完整排空。

原生重放还会在重新发布之前重新校验每一个操作：一个 PROGRAM 必须仍是
一个 `AmdProgram`，其 core 是完全相同的那个 `Arc`（`Arc::ptr_eq`，而不是
某个仅仅报告 `AMD:N` 的分配器），且 PM4 与 AQL 程序地址未变；
一条 COPY 通道则要求已安装 SDMA 队列。

## 后端接缝

KFD 操作被隔离在 `AmdIface` 之后：

```rust
pub trait AmdIface: Send + Sync + std::fmt::Debug {
    fn alloc_raw(
        &self,
        size: usize,
        kind: AllocKind,
        tag: AllocTag,
        cpu_access: bool,
        zero: bool,
    ) -> Result<AllocResult>;
    fn free_raw(&self, gpu_va: u64, size: usize, handle: u64);
    fn setup_ring(&self, desc: &RingDesc) -> Result<QueueHandle>;
    fn teardown_ring(&self, queue_id: u32, doorbell_base: NonNull<u8>) -> Result<QueueTeardown>;
    fn wait_events(&self, timeout_ms: u32) -> Result<Option<Error>>;

    // Defaulted hooks. `KfdIface` overrides the first and the last; the
    // test mock overrides `publication_checkpoint`.
    fn queue_event_mailbox(&self) -> Option<QueueEventMailbox> {
        None
    }
    fn publication_checkpoint(&self, _stage: PublicationStage) -> Result<()> {
        Ok(())
    }
    fn update_queue_percentage(&self, _queue_id: u32, _ring_gpu: u64, _ring_size: u32, _percentage: u32) -> Result<()> {
        Ok(())
    }
}
```

环、GART、EOP、上下文保存以及 inactive-signal 缓冲区都在这道接缝之上分配。
`setup_ring` 激活这些资源并映射 doorbell。
`update_queue_percentage` 正是那个重新映射 AQL 队列、使 CP 固件重新读取其
已缓存的 `amd_queue_t` scratch 描述符的东西——在通道创建时以及每当 scratch
增长时都会这样做；一次失败的重新映射会毒化设备。当存在 `queue_event_mailbox` 时，
每一次完成存储都附带一个 KFD 事件中断伴随操作，使 `wait_events` 无需轮询即可被唤醒。

## 配置

| 变量 | 默认值 | 作用 |
|---|---|---|
| `SVOD_DEVICE` | `CPU`（macOS 上为 `METAL:0`） | 选择默认的张量设备，例如 `AMD:0`（`HIP` 是其别名） |
| `SVOD_AMD_BACKEND` | `kfd` | AMD 后端；目前只接受 `kfd` |
| `SVOD_AMD_HW_QUEUES` | 多 XCC 上为 4，否则为 1 | 有界的计算通道数，夹在 1 到 64 之间 |
| `SVOD_AMD_AQL` | 未设置 | 除 `0` 以外的任何值都会在单 XCC 硬件上强制使用 AQL；在每次创建队列时读取 |
| `SVOD_AMD_SCOPED_SYNC` | 未设置 | `=0` 把每一次以存储为范围的宿主等待都换成一次完整的设备排空，并停止生产者记录；每进程锁定一次 |
| `SVOD_PM4_GRAPH` | 未设置 | `=1` 启用 PM4 图捕获；只有 `1` 算数；在设备创建时读取一次 |
| `AMD_DISABLE_SDMA` | 未设置 | 设成任何值都会跳过 SDMA 复制队列，强制使用宿主可见缓冲区 |
| `SVOD_KFD_TOPOLOGY` | sysfs | 为测试覆盖 KFD 拓扑根目录 |
| `SVOD_DEBUG_DISPATCH` | 未设置 | 设成任何值即打印程序加载以及调度的 grid、kernarg、scratch 与缓冲区地址 |
| `SVOD_DUMP_AMD_IR` | 未设置 | 以 `<mcpu>_<module>.ll` 形式接收每个内核的 AMD LLVM IR 的目录 |
| `SVOD_OBJECT_CACHE`、`SVOD_OBJECT_CACHE_DIR`、`SVOD_OBJECT_CACHE_MAX_BYTES` | 开启 | 共享的磁盘 object 缓存，见 [CPU 页面](../cpu.md) |
| `SVOD_AM_DEBUG` | 未设置 | 仅用于 AM 启动：写入寄存器后再读回 |
| `SVOD_AM_MCBASE` | 未设置 | 仅用于 AM 启动：`raw`、`fb` 或 `fbxgmi` MC aperture 基址 |

不存在 `SVOD_AMD_SINGLE_QUEUE`。当需要单条硬件通道时，
设置 `SVOD_AMD_HW_QUEUES=1`。
