---
sidebar_label: Rangeify 与内核切分
---

# Rangeify、内核切分与预优化

本页的所有内容都在优化器看到内核之前运行。源码：`schedule/src/rangeify/` 以及 `schedule/src/optimizer/mod.rs` 中的 `apply_pre_optimization`。

## Rangeify（`rangeify_with_map`）

输入：一次 `realize()` 调用产生的张量图（movement 算子、`num_axes > 0` 的张量形式 `REDUCE`、`CONTIGUOUS`、`COPY` 等）。输出：每个循环都是显式 `RANGE`、每次物化都是 `STAGE`、每次读取都是 `INDEX` 的图。

各 pass 按顺序如下（`rangeify/transforms.rs`）：

1. **多设备解析** —— 先 `multi_pm` 再 `lower_allreduce_pm`（均为 `graph_rewrite_preserve_calls`）；`validate_supported_subset` 拒绝后端无法运行的内容。
2. **`add_tags_patterns`**（自底向上）为每个可打标签的节点编号 `[i]`。标签就是张量的身份：切分之后，输出映射由幸存下来的标签重建。`PARAM`、`CONST`、`RANGE`、`END`、`CALL`、movement 算子以及全部由 PARAM 构成的 `MSTACK`/`MSELECT` 不打标签。
3. **`resolve_calls`** 用 `FUNCTION` 的函数体替换其参数，并折叠 `GETTUPLE(TUPLE(..), i)`。预编译函数和 `CALL` 参数保持不透明。
4. **最早期重写**（自底向上，一个匹配器）：`movement_op_patterns + early_rewrites + split_reduceop_patterns`。`early_rewrites` 去掉 `DETACH`/`CONTIGUOUS_BACKWARD`，合并未打标签的 `RESHAPE` 链，在加宽类型转换之下加宽整数乘积，用 `CONTIGUOUS` 物化被改变大小/重排的 `COPY` 源，删除同设备的 `COPY`，并把零大小张量折叠为常量。`split_reduceop` 是两阶段归约拆分（参见 [range 优化](../optimizations/range-optimization.md)）。
5. **`run_rangeify`**（`rangeify/indexing.rs`）：
   - `pm_generate_realize_map`（自底向上）：标记必须成为缓冲区的内容 —— `STORE`、`CONTIGUOUS`、`COPY` 及其非连续源、`MSTACK`/`MSELECT` 的源，以及手写内核 `CALL` 的输入（固定为不可移除）。
   - `assign_ranges`：从根到叶的遍历。被物化的节点为每个输出维度获得新的 `Weak` range（`IndexingContext::new_range`；大小为 1 的维度是 `CONST(0)`）。其他节点继承其消费者的 range；当消费者之间不一致时，`merge_consumer_ranges` 要么合并兼容的索引表达式（有效部分被 OR 进 `WHERE(valid, idx, Invalid)`），要么分配新的 range 并标记该轴需要物化。movement 算子通过 `apply_movement_op` 把输出 range 映射为输入 range（`PERMUTE` 对其置换，`EXPAND` 把广播轴置零，`PAD` 把 range 包进有效性 `WHERE`，`RESHAPE` 经由 `apply_reshape_ranges`）。`ending_ranges` 把广播决策反向传播，使得喂给广播的 `REDUCE` 在广播之前被物化（layernorm 的情形）。
   - `apply_rangeify_patterns`（自底向上）：张量形式 `REDUCE` → 循环形式 `REDUCE(src, ranges)`，`num_axes = 0`；`PAD` → `WHERE(valid, src, 0)`；带形状的 `STACK` → 基于其首个 range 的 `WHERE` 链；每个算子的被物化源都被包进 `STAGE` + `INDEX`（`transform_sources_with_bufferize`）；随后删除 movement 算子。类缓冲区的源（`BUFFER`、`PARAM`、`SLICE`、`AFTER` 等）在形状静态时得到单个行主序 `INDEX`（`linearize_static_indices`）；图像和符号形状则对每个坐标保留一个索引。
6. **Mega-pass** —— 对 `symbolic + pm_reduce_simplify + movement_op_patterns + buffer_folding + dead_axis_removal + pm_remove_bufferize` 做一次不动点迭代。这些组相互促进：内联一个 `STAGE` 会暴露出 `symbolic` 能折叠的 range 算术，进而可能让某个 reduce 变得可折叠。具体规则见 [range 优化](../optimizations/range-optimization.md)页面。
7. 基于带标签的反向切片**重建 SINK**：只有携带输出标签的 `STAGE`、`MSTACK`、`CONST`、`PARAM` 和 `AFTER` 节点作为 sink 的源保留下来，并保持原始输出顺序。
8. **缓冲区上限** —— 如果设备报告了 `max_buffers`，`buffer_limit_patterns` 会强制把逐元素源放进全局 `STAGE`，使任何内核都不超过参数数量上限。

对 `[8, 64]` 张量执行 `x.sum(1)` 的结果：

```text
[67] SINK : Scalar(Void)
└── [66] STAGE : Scalar(Float32) shape=[Const(8)]
    ├── [65] CONTIGUOUS : Scalar(Float32) shape=[]
    │   └── [64] REDUCE(Add, num_axes=0, ranges=[27]) : Scalar(Float32) shape=[]
    │       ├── [62] INDEX : Scalar(Float32) shape=[]
    │       │   ├── [11] PARAM(slot=0) : Scalar(Float32) shape=[Const(512)]
    │       │   │   └── [0] CONST(Int(512)) : Scalar(WeakInt) shape=[]
    │       │   └── [55] Add : Scalar(WeakInt) shape=[]
    │       │       ├── [54] Mul : Scalar(WeakInt) shape=[]
    │       │       │   ├── [26] RANGE(U0, Weak) : Scalar(WeakInt) shape=[]
    │       │       │   │   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
    │       │       │   └── [3] CONST(Int(64)) : Scalar(WeakInt) shape=[]
    │       │       └── [27] RANGE(U1, Reduce) : Scalar(WeakInt) shape=[]
    │       │           └── [3] → (see above)
    │       └── [27] → (see above)
    └── [26] → (see above)
```

`U0`/`U1` 是 `AxisId::Unrenumbered`：range 会在切分时按内核重新编号。输入的 `PERMUTE`/`RESHAPE` 已经消失 —— 它们变成了索引表达式 `U0 * 64 + U1`。

## 内核切分（`try_get_kernel_graph`）

首先执行 `kernel_graph_pre_cut`：

- **`pm_add_buffers_patterns`**（自底向上，`RangeifyBufferContext`）：先 `movement_op_patterns`，然后 `flatten_bufferize`（多 range 的 `STAGE` 变成单个扁平 range 加一个还原形状的 `RESHAPE`）、`late_buffer_slice`（DISK 上的 `STAGE(BITCAST|CONTIGUOUS)` 变成 `SLICE`），以及 `bufferize_to_store`。最后这个规则分配一个调度局部的 `BUFFER`（`new_lunique_buffer`，slot 位于高位命名空间），并把 `STAGE(compute, ranges)` 重写为 `AFTER(BUFFER, [END(STORE(INDEX(BUFFER, idx), compute), ranges)])`。`STAGE(AFTER(..))` 复用底层缓冲区；`Local` 的 `STAGE` 保持不动，留给之后的 `pm_add_local_buffers`。已经成形的内核 `SINK`（带有 `KernelInfo` 的）会被屏蔽，重写不会深入其中。
- **`pm_flatten_range`** 对整张图执行一次（自底向上）：根据从源可达的 `RANGE` 重新推导每个 `END`/`REDUCE` 的 range 列表，这样下面的逐内核 pass 就不必重复遍历共享子图。

然后是 **`split_all_stores`**（自底向上）：每个不再有未关闭计算 range 的 `STORE` 或 `END(STORE)` 都变成一个 `CALL`。`split_store` 在内核体上运行 `local_to_param_patterns + rangeify_codegen_patterns`：全局 `BUFFER`/`PARAM` → 代码生成用的 `PARAM(slot)`，由 `LocalAddBufferContext::param_slot` 按匹配顺序编号；`BIND(var, value)` → 该变量，绑定作为 `CALL` 参数保留；`AFTER`/`MSTACK`/`MSELECT` → 其缓冲区；`RANGE(end=0)` → `CONST(0)`；`Unrenumbered` 轴 id → `Renumbered(n)`；`NOOP` → 带类型的零；`CONTIGUOUS` → 其源，并收集 hint。内核体被包进带默认 `KernelInfo` 的 `SINK`；`COPY`/`SLICE` 值仍作为直接的调用体。`Device` range 是“没有未关闭 range”这一条件的唯一例外：它们是启动 lane，会跨越边界保留下来。

最后是 **`validate_normal_kernel_devices`**（每个非拷贝内核只能有一个设备）和 **`fix_assign`**：当内核 B 读取内核 A 写入的缓冲区时，A 的 `AFTER` 被追加到 B 的 `AFTER` 依赖中；出现环则报 `KernelSplitDependencyCycle`。开启 `SVOD_SPEC` 时，`verify_kernel_graph` 会检查结果。

## 逐内核预优化（`apply_pre_optimization`）

在启发式或 BEAM 之前对每个内核体运行，两条路径都会执行（`optimize_kernel_with_config_impl`、`optimize_kernel_beam`、`prepare_scheduler`）。开启 `SVOD_SPEC` 时，会先针对 `spec_tensor` 运行 `type_verify`。

| 步骤 | 匹配器 | 方向 |
|------|---------|-----------|
| movement 算子 | `movement_op_patterns` | 自底向上 |
| load collapse | `pm_load_collapse` | 自顶向下 |
| 拆分 range | `pm_split_ranges + pm_flatten_range`（`SplitRangesContext`） | 自顶向下 |
| 符号化简 | `sym + pm_fold_cast_const + pm_flatten_range` | 自顶向下 |
| 化简 range | `pm_flatten_range + pm_simplify_ranges`（`SimplifyRangesContext`） | 自顶向下 |

**`movement_op_patterns`** 有三条规则：`INDEX(mop(x), idx)` → `INDEX(x, mop⁻¹(idx))`（`transform_movement_through_index`）、`AFTER(mop(x) | INDEX(x), deps)` → `mop(AFTER(x, deps))`（`push_op_through_after`），以及 `END(mop(x), ranges)` → `END(x, ranges)`。`is_movement()` 恰好是 `RESHAPE`、`PERMUTE`、`EXPAND`、`PAD`、`SHRINK`、`FLIP`。它以自底向上方式应用，因为内层 movement 算子必须先被重写，其消费者才能匹配。

**`pm_load_collapse`** 消除这样的 `REDUCE(Add)`：经过符号推理后其主体与 range 无关（`reduce_load_collapse`）。reduce 作用域之外的节点被替换为标量 `PARAM` 变量（`UOp::variable("in{n}", vmin, vmax)`），主体被包进一个只覆盖该 range 的合成 `REDUCE`，运行 `build_reduce_load_collapse_matcher`，如果没有 `RANGE` 幸存，就撤销替换。它使用的界限模式见 [range 优化](../optimizations/range-optimization.md)页面。

**`pm_split_ranges`** 记录每个其 end 能被常量整除的 `RANGE % const`（排除 `Warp` 和 `Device` range；图像 `STORE` 索引到的每个 range 都被固定），并在 `SINK` 处一次性替换 `r → outer * c + inner`，轴 id 为 `r.child(0)` / `r.child(1)`。替换后的图再用 `symbolic + pm_fold_cast_const` 化简。

**`sym`** 是完整的第 3 层化简器（[代数化简](../optimizations/algebraic-simplification.md)）；`pm_fold_cast_const` 折叠 `CAST(CONST)`；`pm_flatten_range` 在 range 消失后保持 range 列表的准确。

**`pm_simplify_ranges`** 在合并后的形式不增加 `FloorDiv`/`FloorMod` 数量时，合并某个 `END`/`REDUCE` 的相邻 range（`simplify_merge_adjacent`），并把一个 range 收窄到任意 `INDEX` gate 能为其证明的最大界限（`mark_gated`；只要有一处未加 gate 的使用就会固定原始 end；`REDUCE` 的 range 受保护）。两种替换都在 `SINK` 处进行。

## 交给优化器

`Scheduler::new(ast, renderer)` 收集 extent > 1 的 `RANGE`，按 `(axis_type.priority(), axis_id)` 排序；在具备 `has_local` 的 renderer 上，`convert_loop_to_global` 把 `Weak` 输出轴变成 `Global`（在 CPU 上它是空操作，这就是上例中行轴仍为 `Weak` 的原因）。之后 `hand_coded_optimizations` 或 BEAM 应用若干 `Opt`，`get_optimized_ast_with_naming` 输出带 `KernelInfo` 元数据的内核 `SINK`（名称如 `r_8_16_4`、`dont_use_locals`、`opts_to_apply`）。`SVOD_NOOPT` 会跳过启发式，但不会跳过本页内容，也不会跳过 post-optimization 各阶段。搜索本身在[内核搜索](../optimizations/kernel-search.md)中介绍。
