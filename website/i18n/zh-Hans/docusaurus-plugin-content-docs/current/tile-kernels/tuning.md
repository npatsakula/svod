---
sidebar_label: 自动调优
---

# 首次使用时自动调优

内核的分块表是一个搜索空间，而不是答案。GEMM 每个架构家族带三到五个分块，flash attention 带四种每 warp 分块，单查询注意力带若干种 K/V 切分——哪一个胜出取决于形状、设备的计算单元数及其时钟。因此，设备第一次遇到某个形状时，`svod-tk` 会编译并计时每个适配的候选，保留最快的那个，并把它记在磁盘上。整套机制就是 `tk/src/tune.rs`。

---

## 首次启动时发生了什么

`GemmPolicy::tuned`、`FaPolicy::tuned` 和 `SqPolicy::tuned`（位于 `tk/src/kernels/`）都通过 `TuneStore::select` 走同一套流程：

1. **过滤**：把表缩减到能对该形状分块（对 GEMM 而言，还要带有所请求的 `Epilogue`）的候选。若只剩一个或一个都没有：直接返回静态选择，不做任何测量。
2. **内存缓存（memo）**：一个进程级的 `HashMap<TuneKey, usize>` 无需构建内核即可回答重复的形状——一个 plan 对每个节点都会问一次同样的形状。
3. **存储**：memo 未命中时，针对占位缓冲区构建每个候选的 `SINK` 并计算指纹（`kernel_fingerprint`）。这些摘要会并入存储行，因此内核函数体一改就会重新测量。然后在该设备的文件中查找这一行。
4. **测量**：存储未命中时，在该形状的合成操作数上（`Tensor::randn`，转换为目标 dtype，移到设备上）对每个候选执行 `compile_kernel`。第一个构建成功的候选负责把时钟拉起来——`warm_clock` 反复派发它，直到耗时不再下降或已过去 1.5 s；已在负载下的设备几次运行就会进入平台期。随后 `round_robin_min` 轮流对每个候选计时三轮，并保留各自的最小值，这样不会有哪个候选是在其他候选没遇到过的时钟下被评判的。
5. **保留**最快的那个：写入 memo，也写入存储文件。无法构建或派发的候选会被跳过；若全部失败，则不缓存任何东西，改用静态选择。

只有测出的胜者才会写入文件。`select_with` 是同一套策略，只是测量由调用方提供，`tk/src/test/unit/tune.rs` 中的单元测试正是借此在没有 GPU 的情况下检验它。

---

## 键与存储

```rust
// tk/src/tune.rs
pub struct TuneKey {
    pub kernel: &'static str,   // "gemm_nt", "flash_attention", "sq_attention"
    pub device: String,         // "<arch target name>-<compute units>cu"
    pub shape: Vec<usize>,      // the kernel's own shape tuple, dtype width and flags included
    pub config: u64,            // a digest of the candidate set (and anything else the graphs vary with)
}
```

GEMM 以 `[m, k, n, dtype.bytes(), epilogue.code()]` 为键；flash attention 以 `[b, n, h, h_kv, d, causal, mask.code(), dtype.bytes()]` 为键。改动分块表会改变 `config`，因此新增的候选会触发重新测量。

存储是每个设备、每个 crate 版本一个文件，每个条目一行：

```text
<kernel>|<device>|<shape>|<builds digest> <winning index> <ns>
```

文件位于以下位置中第一个满足条件的：

| 位置 | 条件 |
|---|---|
| `$SVOD_TK_TUNE_DIR/` | 设置了该变量 |
| `$XDG_CACHE_HOME/svod/tk_tune/` | 否则，若设置了 `XDG_CACHE_HOME` |
| `$HOME/.cache/svod/tk_tune/` | 其他情况 |

文件名是设备字符串，其中非字母数字字符被替换（`gfx1201_64cu-v0.1.0.txt`）。写入时会重新读取、合并，并以原子方式重命名，因此两个进程同时调优时，最多丢失对方最新的那一行。目录不可读或不可写只算未命中，绝不报错；若没有可写的根目录，存储就只在内存中。

---

## 关闭调优

| 控制方式 | 效果 |
|---|---|
| `SVOD_TK_TUNE=0` | 不做测量；每个策略都返回其静态选择（`GemmPolicy::cfg`、`FaPolicy::config`、策略自身的切分） |
| `svod_tk::tune::set_enabled(false)` | 效果相同，但在代码中设置，并在整个进程内覆盖环境变量——测试框架会调用它，以免一个内核测试对它碰到的每个形状都做调优 |
| `gemm_nt_with(x, w, cfg)`、`flash_attention_tuned(q, k, v, opts, policy)`、`SqAttentionOpts::split` | 用你自己的选择器在单次启动中绕过策略 |

当设备不打派发时间戳（`dispatch_gpu_ns` 为 `None`）时，调优也会被跳过，因为没有可比较的东西。

---

## 调优的对象

| 内核 | 候选 | 分块表 |
|---|---|---|
| `gemm_nt` | 该架构家族分块表中能对 `(m, k, n)` 分块且带有该尾声的每个 `GemmCfg` | `tk/src/kernels/gemm.rs` 中的 `CUDA_TILES` (2)、`RDNA_TILES` (3)、`RDNA4_TILES` (5) |
| `flash_attention` | `FA_TILES` 中 K/V 双缓冲能放进共享内存、且块大小能整除 `N` 的每个 `(q_blk, kv_blk)` | `tk/src/kernels/fa.rs` 中的 `[(16,16), (16,32), (16,64), (32,32)]` |
| `single_query_attention` | `N` 的约数中最接近设备驻留 wave 预算、且每块至少留下 15 次迭代的那些 | `tk/src/kernels/sq_attention.rs` 中的 `SqPolicy::candidates` |

每个策略回退时采用的静态选择本身也是测量得来的，每个架构家族在一块硬件上测过：GEMM 的 `GemmPolicy::cfg` 优先选最宽的分块，除非它的网格无法把设备的计算单元填满 `resident` 倍；`FaPolicy::tile` 只有在启动网格覆盖整台设备、且头维度低于该家族的上限时，才选更高的每 warp 分块。调优之所以存在，是因为这些交叉点会随形状移动。

:::tip[解读一次测量]
`SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib tune::gemm_first_use -- --ignored` 会针对一个临时存储运行真实的流程，并断言：只有一个文件、一行记录，且第二次请求无需测量就能读回它。要查看你自己的运行中某个形状选了什么，直接读那个文件即可：索引是上面所列分块表中的位置。
:::

这笔开销对每个设备上的每个形状只付一次：几次编译，加上冷 GPU 大约两秒的计时。形状经过分桶的模型——Qwen3 把序列长度分桶到 `FLASH_ATTENTION_SEQUENCE_MULTIPLE`——只需调优寥寥几行，之后每个批次都直接走 memo。
