---
sidebar_label: 内核来源
---

# 内核来源

profile 报告 `r_128_3_32_4_2_2_2_4_4_192_2` 耗时 100 ms，说明的只是内核的形状，而不是它归谁所有。来源回答的正是后一个问题：凡是 dispatch 出去的内核，都知道自己是为哪个模块路径、哪个调用点或哪个 ONNX 节点而构建，profiler 也就能沿着这条路径把时间汇总起来——按层、按块、按阶段。

本页是使用指南：如何开启、如何给模型加标注、如何读懂输出。其机制（每个节点上一个参与 hash-cons 的字段，到内核切分处再被剥除）在文末有简要说明，完整文档见 [IR 设计](./ir-design.md)与[操作图鉴](./op-bestiary.md)两页。

---

## 开启 {#turning-it-on}

捕获默认关闭，关闭时零开销：节点不携带来源，哈希与不含该特性的构建逐字节相同。两个开关：

| 开关 | 作用 |
|--------|--------|
| `SVOD_ORIGIN=1` | 对进程的每个线程开启捕获（未设置、为空或 `0` = 关闭） |
| `SVOD_ORIGIN_DEPTH=<n>` | 汇总保留前 `n` 个路径段（未设置或 `0` = 完整路径，打印为 `depth leaf`） |

```bash
SVOD_DEVICE=AMD:0 SVOD_ORIGIN=1 cargo run --release -p svod-model --example gigaam_infer -- \
    audio.wav --profile --origin-depth 3 --profile-json profile.json
```

`--origin-depth` 优先于 `SVOD_ORIGIN_DEPTH`；Whisper 示例只有 `--profile`。

测试里只为当前线程切换捕获，好让并行的测试各自保持图身份：

```rust
let _capture = svod_ir::origin::capture_for_thread(true); // restored on drop
```

---

## 来源从哪里来 {#where-origins-come-from}

来源是一条帧路径，根在最前。每个帧是下列之一：

| 帧 | 渲染为 | 由谁打开 |
|-------|-------------|-----------|
| `Module` | `encoder.layers.3.ffn1` | 模型代码，每个模块一段（`OriginScope::module`） |
| `Label` | `vad`、`GigaAmCtcJit`、`initializer` | 流水线阶段、每次 `jit_wrapper!` 构建、ONNX 导入器 |
| `Onnx` | `/encoder/Conv` 或 `#12:MatMul` | ONNX 导入器，每个节点一个：取节点名，没有名字时取 `#index:op_type` |
| `Call` | `@ linear model/src/gigaam/encoder.rs:43` | 每个公开的 `Tensor` 操作，在入口处通过 `origin_call!` 打开 |

具名帧之间用 `.` 连接；`Call` 帧跟在一个空格之后。它是模块路径之下扁平的一层 file:line：只有当前帧还不是调用帧时，操作才会打开它（以最外层为准），因此建立在其他操作之上的操作（`linear` 之于 `matmul`）只记下用户那一行一次，绝不会记成 svod 自己的源码。它上面的模块层则由模型代码添加。

### 为 Rust 模型加标注 {#instrumenting-a-rust-model}

在 `forward` 里为每个模块打开一个作用域，名字照着它的 state-dict 前缀来取。模型 crate 里有恰好做这件事的辅助函数：

```rust
use crate::state::{scoped, scoped_index};

fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let x = scoped("subsampling", || self.subsampling.forward(x))?;
    let mut x = x;
    for (i, layer) in self.layers.iter().enumerate() {
        x = scoped_index("layers", i, || layer.forward(&x))?;   // layers.0, layers.1, …
    }
    scoped("final_norm", || self.final_norm.forward(&x))
}
```

`scoped` 在闭包外围打开 `OriginScope::module(name)`；`scoped_index` 只在捕获开启时才格式化 `name.i` 段；`scope_index(name, i)` 与之相同，只是以 guard 的形式提供，用于无法写成闭包的循环体。每个模块只打开自己那一段，嵌套会重建出完整路径，于是 profile 打印的路径就等于它所触及权重的 state-dict 键前缀。GigaAM 和 Whisper 都是这样加标注的，`model/src/test/unit/origin.rs` 会断言这两组路径一致。

流水线阶段是根段。GigaAM 把 JIT 构建包在一个模块作用域里，`arch` 流水线则使用标签——无论哪种方式，阶段名都会出现在其内部生成的每条路径开头：

```rust
scoped("ctc_head", || jit.prepare_with_config(mel_spec, lengths_spec, &config))?;

let _stage = OriginScope::label(self.profile_label());   // "vad"
self.vad.probs(waveform)?;
```

`jit_wrapper!` 宏会在每个构建闭包外围打开 `OriginScope::label("<WrapperName>")`，下文的 `GigaAmCtcJit` 段就由此而来。在任何作用域之外构建的东西都会落到 `<unattributed>` 行。

### ONNX 图 {#onnx-graphs}

无需任何操作。导入器为每个节点打开一个 `Onnx` 帧（索引、名称、op 类型、domain、opset），并在拥有子图分支的节点之下为每个分支（`then_branch`、`else_branch`）打开一个 `Label`，因此 `If` 的分支体读作 `#7:If.then_branch.#0:Add`。初始化器和图输入位于 `initializer` 与 `input` 之下——但仅当它们构建出计算节点时如此：字面量和缓冲区从不携带来源。

### 手写内核 {#hand-written-kernels}

`tk` 内核按其构建时活跃的作用域归属——与图内核规则相同。调度器从不看到它的内核体，因此 `UOp::custom_kernel` 在构造时就收集并剥除来源（`ir/src/uop/constructors/graph.rs`）；两个层启动同一个手写内核，仍共享同一个编译后的程序。它拷入的输入继承其生产者的来源。

---

## 读懂输出 {#reading-the-output}

开启捕获后，`--profile` 先打印常规的逐内核表，然后是两份汇总。示例为 GigaAM v3 编码器，f16，在 gfx1151 上处理一个 60 s 窗口，深度截为 3：

```
519 dispatches (519 GPU-stamped), total 444.237 ms
  total ms  count    mean µs      %  name
   103.183     16     6448.9   23.2  r_128_3_32_4_2_2_2_4_4_192_2n1
   100.305     16     6269.1   22.6  r_128_3_32_4_2_2_2_4_4_192_2
    80.530     32     2516.6   18.1  r_128_12_32_4_2_2_2_4_4_48_2
    …
origin rollup (depth 3, exclusive; rows sum to the total):
  total ms  count    mean µs      %  origin path
    27.833     32      869.8    6.3  ctc_head.GigaAmCtcJit.layers.3
    27.678     32      864.9    6.2  ctc_head.GigaAmCtcJit.layers.9
    27.620     32      863.1    6.2  ctc_head.GigaAmCtcJit.layers.0
    …
    23.334      2    11666.8    5.3  ctc_head.GigaAmCtcJit.subsampling
     0.661      4      165.2    0.1  ctc_head.GigaAmCtcJit.head
     0.131      1      131.0    0.0  ctc_head.GigaAmCtcJit
origin rollup (depth 3, inclusive; parents contain children, rows overlap):
  total ms  count    mean µs      %  origin path
   444.237    519      855.9  100.0  ctc_head
   444.237    519      855.9  100.0  ctc_head.GigaAmCtcJit
    27.833     32      869.8    6.3  ctc_head.GigaAmCtcJit.layers.3
    …
```

读法：

- **Exclusive（独占）** 把每次 dispatch 只计一次，记到它的*主*来源上：即产生该内核所存储值的那个作用域（若该值不带来源，则取根之下最近的有归属节点）。各行构成总量的划分，因此十六个 `layers.N` 行加上 `subsampling`、`head` 以及剩余的 `GigaAmCtcJit` 行，合计恰为 519 次 dispatch、444 ms。十六层、每层 32 次 dispatch，就是整个编码器；各层之间的差异（25.3 到 27.8 ms）是真实存在的，也是你首先该看的地方。独占行少于十二行时，每行还会以 `· name` 行列出其前三个内核。
- **Inclusive（包含）** 把一次 dispatch 记到融合进它的每个来源的每个祖先上。父行包含其子行，所以 `ctc_head` 为 100 %，各行彼此重叠。用它来查看一个块有多少时间藏在跨模块边界融合的内核里。
- **Depth（深度）** 是保留的路径段数。这里深度 3 给出逐层的行；深度 4 会把一层拆成 `ffn1`、`mhsa`、`conv`、`ffn2`、`final_norm`；leaf 保留完整路径。`Call` 帧从不构成汇总键——它们只是内核行和 JSON 中的细节。
- 融合了两个模块的内核，独占地记到它所存储值的那个模块上（残差加法记在层上，而不是 `ffn2` 上），包含地则记到两者上。

任何 `RunProfile` 都同时带有两份汇总：`render_report()`（上面的布局，`gigaam_infer` 使用）和 `render_table()`（Whisper 的 `--profile`，是另一种内核表）都会追加同样的来源部分，按生成 profile 时的深度截断；`render_report_at(d)` / `render_table_at(d)` / `to_json_at(d)` 可覆盖该深度。

### JSON {#json}

`--profile-json out.json`（或 `RunProfile::to_json()`）每次运行写出一个文档（节选）：

```json
{
  "origin_depth": 3,
  "stages": [{
    "name": "ctc_head", "wall_ms": 463.8, "gpu_ms": 444.2, "dispatches": 519, "meta": {},
    "kernels": [{
      "name": "r_128_3_32_4_2_2_2_4_4_192_2", "count": 1, "total_ms": 6.3, "mean_us": 6269.1,
      "origin": "ctc_head.GigaAmCtcJit.layers.3 @ add model/src/gigaam/encoder.rs:596",
      "origin_id": 41, "origins": ["…"], "origin_ids": [41, 39]
    }],
    "origins_exclusive": [{ "path": "ctc_head.GigaAmCtcJit.layers.3", "count": 32, "total_ms": 27.8, "mean_us": 869.8, "percent": 6.3, "kernels": [] }],
    "origins_inclusive": []
  }],
  "origins": [{ "id": 41, "parent": 40, "frame": { "Module": { "name": "layers.3" } } }]
}
```

内核行以入口点*和*主来源共同作为键，因此同一个程序会按派发它的每个作用域各出现一次。`origins` 只包含本次运行引用到的帧，并在 `parent` 下闭合，所以无需写出该文件的进程也能解析 id。

---

## 线程 {#threads}

捕获状态是逐线程的：开关、当前作用域，以及该作用域是否为调用帧。作用域不会跟随工作转移到其他线程；作用域 guard 是 `!Send` 的，并恢复它被打开时所在的线程。由此得出的规则：

- 在打开作用域的那个线程上构建图。GigaAM 和 Whisper 都是这样做的；在 `prepare_with_config` 外围打开的阶段作用域覆盖其内部构建的一切。
- 调度和编译以分离模式运行（`OriginScope::suspend`），在调用方线程和 rayon worker 上都是如此，因此环境作用域绝不会泄漏进内核体；到那时归属早已收集到 CALL 上了。
- 若要把作用域带到你自己派生的 worker 上，先捕获 `origin::current()`（一个 `Option<OriginId>`），再在那里用 `origin::install(id)` 重新安装。worker 和其他线程一样，从 `SVOD_ORIGIN` 初始化自己的开关。
- BEAM 搜索在子进程中、针对不含来源的内核体运行；它从不看到作用域。
- **异步代码：** 作用域必须嵌套，所以不要跨 `.await` 持有作用域。打开作用域，同步地构建图，释放它，然后再 await。guard 是 `!Send` 的，因此跨 await 保持 guard 存活的 future 无法在多线程执行器上被 spawn；而当两个任务在同一线程上交错使用作用域时（一个 guard 在后打开的 guard 仍活跃时被释放），debug 构建会 panic。svod 中的图构建是同步的，所以代码的自然形态本就满足这一点。

---

## 成本与权衡 {#costs-and-trade-offs}

- **关闭时：** 没有任何开销。每个节点一次线程局部读取，无分配，哈希不变。
- **开启时：** 每次进入作用域做一次 interning（arena 上的一把互斥锁，每次 forward 几百次），每个公开操作为调用帧做一次线程局部写入，以及在切分处每个内核一次拓扑排序以收集并集。开启和关闭捕获时，GigaAM 的 dispatch 次数与 GPU 时间完全相同。
- **身份会改变。** 来源是节点身份的一部分，因此在不同作用域下构建的两个相同表达式，在切分剥除来源之前是两个节点。内核程序不受影响——剥除会恢复去重——但若某个辅助函数为每个调用点重建同一表达式（掩码 clamp、表的 cast、输入拷贝），它就会按作用域各物化一次。请在 `OriginScope::suspend()` 下运行这类辅助函数，或让拷贝继承其生产者的来源；`custom_kernel` 对其输入已经采用了后一种做法。出于同样的原因，常量、缓冲区、参数、`UNIQUE`、`DEFINE_VAR`、`BIND`、`STACK` 以及所有 `Index` 类型的节点从不携带来源。
- **依赖结构身份的测试**（期望两个手工构建的图 hash-cons 成同一个节点）应在 `capture_for_thread(false)` 下运行。

---

## 一段话讲清工作原理 {#how-it-works-in-one-paragraph}

作用域活跃期间构建的每个 `UOp` 都存储该作用域 4 字节的 `OriginId`（一个 `NonZeroU32`，因此 `Option` 仍为四字节），并将其折入内容哈希，于是来自不同作用域的相同子图在 rangeify 过程中保持区分。在内核切分处，`split_store` 遍历内核体一次，取被存储值的来源作为主来源、取并集作为集合，把两者都记到内核 `CALL` 的 `CallInfo` 上，并以清空来源的方式重建内核体。切分之后的一切——优化器、BEAM、代码生成、每一个内核缓存——看到的都是不含来源的 AST。执行计划把 CALL 的归属复制到每个准备好的操作上，profiler 复制到每个 `KernelProfile` 上，汇总则把父链截断到所请求的深度。
