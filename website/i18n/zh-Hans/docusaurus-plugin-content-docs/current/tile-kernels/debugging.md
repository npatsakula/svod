---
sidebar_label: 调试
---

# 调试与验证内核

一个手写内核有多可信，取决于你检查它的能力有多强。[Flash Attention](./flash-attention) 的演练展示了什么样的内核值得手写；本章则讲你如何一步步把它信下来。USE 面孔交给你的是一个融入大图的惰性 `Tensor`，方便归方便，可要在这里问「这一个内核对不对、有多快」却很别扭。`tk` 的 **DEBUG 面孔**正是为此而生：让单个内核针对具体缓冲区运行，把结果读回来，给它计时，并证明一次重构没改变它的行为。

---

## 直接派发：跑一个内核，把字节看个清楚

直接启动 API（`tk/src/launch.rs`）完全绕开张量调度器。你给它一个内核函数体和真实的输入张量，它便实现输入、分配输出、渲染、编译、派发，把结果写进一个你能读回的输出：

```rust
// The DEBUG face from tk/src/lib.rs. `outs` are written in place.
run_kernel("tile_add", [1, 1, 1], block, &mut [&mut out], &[&input_a, &input_b], build)?;
let values = out.as_vec::<f32>()?;   // read the GPU result straight back
assert_eq!(values, expected);
```

因为这跳过了调度、融合和依赖跟踪，你测到的*就只是你的内核*，而不是一张恰好包含它的图。这份隔离正是要点：数字一旦错了，你想确切知道它就错在*这里*，而不是错在某条融合流水线里的某个角落。

关于这条路径多说一句：跳过*调度器*并不等于跳过*优化器*。`compile` 仍会在你的 `SINK` 上跑生产用的 `optimize_kernel_with_config`——它对一个手工降级的函数体不施加任何调度优化（这正是 `opts_to_apply: Some(vec![])` 这个标记换来的），但依然会执行每个内核在渲染前都需要的那些共享重写，其中就包括索引 dtype 的降级。于是你不靠调度器也拿到了正确的代码。`ArchCaps` 取自缓冲区所在的设备；架构无法解析的 GPU 会直接报错，只有主机设备才会回退到 `ArchCaps::GFX942`，好让 `SINK` 依然能构建出来。

---

## 在真实硬件上计时

做性能工作时，`CompiledLaunch`（来自 `compile_kernel`）暴露的是硬件时间戳，而非挂钟上的估摸：

```rust
// Render + compile once …
let launch = compile_kernel("matmul", grid, block, &mut [&mut c], &[&a, &b], build)?;
// … then dispatch in a loop, outside the timed region.
// SAFETY: the bound buffers stay allocated for `launch`'s lifetime.
unsafe { launch.dispatch(true) }?;
let ns = launch.dispatch_gpu_ns()?;   // Option<u64>: device-measured dispatch time
```

`dispatch_gpu_ns()` 通过一个剖析上下文派发一次，并在前后读取设备自己的时间戳计数器，所以你测到的是设备上的时间，而不是启动它那一来一回的延迟——在不打任何时间戳的后端上返回 `None`。[自动调优器](./tuning) 正是用这个原语给候选排名的，排名前先用基准测试同样使用的 `warm_clock` 把时钟拉起来。criterion 基准则在上一层、经由 `plan.profile` 拿到同一批时间戳；参见 [剖析与基准测试](./profiling)。

---

## 无 GPU 的测试与有 GPU 的测试

构建 `SINK` 只是纯粹的 UOp 构造，不需要设备；只有执行它才需要。测试模块（`tk/src/test/unit/`）处处都利用这一划分，新内核也应如此：

- **图形状测试**在每次 `cargo test` 时运行。针对占位缓冲区（`UOp::new_buffer(DeviceSpec::Cpu, size, dtype)`、`ArchCaps::GFX942`）构建内核，对 `SINK` 做拓扑排序，然后断言其中有什么、没有什么——`guide.rs` 检查 tile-add 内核生成了一个 `Op::Special`、恰好含一个 `Binary(Add)`，且没有 `Wmma` 也没有 `Local` 缓冲区。
- **硬件测试**标记为 `#[ignore]`，并在不受支持的设备上自行跳过：

```bash
SVOD_DEVICE=AMD:0  cargo test -p svod-tk --lib guide::test_tile_add_amd -- --ignored
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk --lib fa::test_fa_graph_check -- --ignored --nocapture
```

门控位于 `tk/src/test/unit/mod.rs`：`device_supported(archs)` 对应内核的 `ArchSet`，`fragment_device()` 用于任何需要矩阵核心布局的场合，`is_cdna_device()` 和 `wave32_fragment_device()` 用于特定布局的检查。它们每一个都会调用 `svod_tk::tune::set_enabled(false)`，以免一个数值测试对它碰到的每个形状都做调优。

对于图原生内核，`svod_tensor::custom_kernel_check!` 会生成整套比较：给定形状和 dtype 的随机输入、被测内核、一个参考闭包，两者都转换为 f32，并以 `atol = rtol = tol` 比较。

```rust
svod_tensor::custom_kernel_check! {
    test_fa_graph_check,
    inputs (q, k, v): shape [1, 128, 2, 64], dtype svod_dtype::DType::BFloat16,
    run: |q, k, v| {
        let out = crate::kernels::fa::flash_attention(q, k, v).expect("FA build");
        Ok::<_, crate::LaunchError>(out.expect("the FA kernel applies to [1, 128, 2, 64] bf16 on every supported arch"))
    },
    reference: fa_causal_reference,
    tol: 2e-2,
}
```

在这里，拒绝（`Ok(None)`）会明确地失败，而不是拿参考实现跟它自己比较。

---

## 指纹：证明一次重构保留了行为

手写内核有个微妙的风险：你「整理」了一下构建器代码，内核照样能编译、照样产出看似合理的数字，但*生成的 IR* 却以某种只在日后某个形状、某个架构上才暴露的方式变了。

`KernelFingerprint`（`tk/src/fingerprint.rs`）就是防这一手的。LLVM 渲染结果在不同运行之间并不确定（节点 id 会泄漏进 SSA 名称），但*图*是确定的：每个 UOp 都带有一个递归的结构化 `content_hash`，指纹就是 `SINK` 的这个哈希，再加上对节点标签的一次与顺序无关的折叠——一个 `u128` 的 `digest`，外加便于阅读差异的 `op_counts` 和 `node_count`。

```rust
let fp = kernel_fingerprint(&sink);
assert_eq!(fp.digest, GOLDEN_MATMUL_DIGEST);  // structure unchanged ⇒ behavior unchanged
```

指纹一旦挪位，就说明你改了所发射的 IR（无论有意无意），金标准测试会逼你正视它。`tk/src/test/unit/golden.rs` 正是用这一招锁住了 matmul 和 flash-attention 的构建器（因果、非因果、带掩码）；失败时会打印出可供粘贴的新摘要，而一次有意的重新定基需要通过导出并 diff 两张图来证明。同样的摘要也是 [自动调优器](./tuning) 磁盘存储的键，因此内核一改就会重新测量它的分块。

---

## 不报错的错误

分块内核是一张依赖图，缺一条边得到的是错误的答案，而不是编译错误。测试套件抓到过的有这些：

| 症状 | 原因 | 修复 |
|---|---|---|
| 累加器跨循环迭代带着陈旧状态 | 每次迭代的重新初始化（`g.zero(acc)`）不依赖循环计数器，被提升到了循环之上 | `g.zero(lp.reinit(acc))` |
| 循环携带的分块在循环结束后读到的是循环前的值 | 最终读取没有排在循环的 `END` 之后 | `acc.after(&lp.close())` 或 `lp.close_carry(acc)` |
| `finish` 触发调试断言，或线性化器把循环的作用域搞错 | 两个存储 `END` 了同一个 `RANGE` | 每个循环只有一个收尾存储；其他存储串接到它上面 |
| 缓冲区不对，数量却对 | `gl` / `bind_abi` 的顺序与启动时的 `[outs..., ins...]` 不同 | 先声明输出，再按启动顺序声明输入，可选缓冲区放在最后 |
| 内核悄悄读错了 K/V 流 | `k`/`v` 以 `q` 的形状绑定，但 dtype 不同而宽度相同（`Kernel::gl` 只检查字节宽度） | 在 `validate` 中校验 dtype，就像 `flash_attention_with` 那样 |
| 在 CDNA 上正确，在 RDNA 上是垃圾 | 硬编码了 lane 数或 fragment 常量 | 读取 `caps.wave_size` 和 `ker.frag(role)`（[布局与 wave 宽度](./wave-portability)） |

*会*报错的是构建器的断言：分块维度不是其 fragment 的倍数、`k_step` 不是矩阵核心 K 边长的倍数、块不是整数个 wave、多 wave 组调用了单 wave 操作、在没有布局的架构上调用 `ker.frag`。每条断言都会指明出问题的值。

---

## 哪个问题用哪个工具

| 你在问…… | 用 |
|----------------|-----|
| 「这个构建器发射的还是我以为的那样吗？」 | 对 `SINK` 拓扑排序结果的图形状测试 |
| 「这个内核产出的数字对吗？」 | `run_kernel` + `as_vec`，或用 `custom_kernel_check!` 与参考实现比较 |
| 「它在这块 GPU 上有多快？」 | `compile_kernel` + `dispatch_gpu_ns` |
| 「我的重构改动了所发射的 IR 吗？」 | `KernelFingerprint` 金标准测试 |
| 「调优器选了哪个分块，为什么？」 | `SVOD_TK_TUNE_DIR` 下的存储文件（[自动调优](./tuning)） |
| 「是*设备/驱动层*在捣乱吗？」 | [AMD 后端 → 调试](../backends/amd/debugging)、[CUDA 后端 → 调试](../backends/cuda/debugging) |

最后一行很重要：本章讲的是调试*内核*，也就是你编写的 IR 和它产出的数字。当问题落在那一层之下时（队列派发、内存故障、驱动、PTX JIT），各后端自己的章节才是该去的地方：[AMD](../backends/amd/debugging) 与 [CUDA](../backends/cuda/debugging)。

---

## 为什么这很重要

手工编写，是拿优化器的安全网去换控制权。DEBUG 面孔就是你安全地做这笔交换的途径：用隔离来定位正确性 bug，用硬件时间戳来撑起经得起推敲的性能论断，用结构化指纹让「我只是整理了一下代码」不会悄悄变成「我改了内核」。有这三样在手，手写内核就和自动调优内核一样可验证。
