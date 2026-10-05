---
sidebar_label: Expander 与归约
---

# Expander 与归约降级（阶段 08–11）

最初的四个 post-optimization 阶段接收优化器的输出 —— 一个 `RANGE` 已带有 `Upcast`/`Unroll`/`Global`/`Local`/`GroupReduce` 轴类型的内核 —— 并把意图具体化：展开的 range 变成带形状的常量，`REDUCE` 变成累加器循环，local `STAGE` 变成 local 缓冲区。它们都在 `apply_post_optimization_configured_with_capture`（`optimizer/mod.rs`）内部运行。

## 08 —— post-opt 符号化简

`POST_OPT_SYM = sym() + pm_move_where_on_load() + pm_flatten_range() + pm_reduce_unparented()`，一次自顶向下的不动点迭代。源码顺序很重要：后面的组会消费前面的组产生的结果。

- `sym()` 是完整的第 3 层化简器（[代数化简](../optimizations/algebraic-simplification.md)）。
- `pm_move_where_on_load`（`symbolic/patterns.rs`）把 `WHERE(cond, INDEX(buf, idx), 0)` 重写为 `INDEX(buf, WHERE(cond', idx, Invalid))`。条件按 `AND` 拆分；只有当某个子句的所有 range 都在该 `INDEX` 的作用域内、且它自身不依赖 `INDEX` 时，该子句才会移入索引；其余子句留在外层 `WHERE` 中。反向形式 `WHERE(cond, 0, INDEX(..))` 用取反后的条件处理。此后有效性藏在索引表达式内部，devectorizer 和 `indexing_simplify` 都能看到它；直到 `19e` 它才变成 LOAD/STORE 的 `gate`。
- `pm_flatten_range` 重建 `END`/`REDUCE` 的 range 列表。
- `pm_reduce_unparented` 丢弃主体未引用的 reduce range：`Add` 乘以 extent，`Mul` 取 extent 次幂，`Max` 直接丢弃该 range（没有 `Min` 分支；`Min` 归约不会被匹配）。

## 09 —— expander（`pre_expand`）

`expander2() + pm_flatten_range() + mop_cleanup_patterns()`，使用 `RangeMap` 上下文（`expand.rs`）。`build_range_map` 按拓扑排序为每个 `Upcast`/`Unroll` `RANGE` 分配一个坐标位置；该映射的长度就是本阶段创建的带形状值的秩。

三条规则，按源码顺序：

| 规则 | 作用 |
|------|--------|
| `Reduce { .. }` → `expand_reduce` | 若循环形式 `REDUCE` 的 range 列表中含有带形状的非 `RANGE` 条目，这些条目的轴（extent > 1）会变成前置的*水平*轴：源被置换使它们排在最前，`num_axes` 统计它们的数量；结果被 reshape 以保留大小为 1 的占位维度。 |
| `Range { axis_type: Upcast \| Unroll }` → `expand_range` | 该 range 变成 `RESHAPE(STACK(CONST(0), ..., CONST(end-1)), shape)`，其中 `shape` 除该 range 自身的坐标外全为 1。该 range 的每个消费者都通过广播变成带形状的值；此时还没有任何复制。 |
| `Wmma { metadata.upcast_axes: Some(..) }` → `expand_wmma` | `contract_axis` 把 A/B 的 upcast 坐标移到末尾并展平进 fragment 操作数；`unroll_axis` 在输出上恢复 C 的坐标。元数据中的 `upcast_axes` 被清空。 |

`mop_cleanup_patterns`（`devectorize.rs`）即 Tinygrad 的 `mop_cleanup`：合并嵌套的 `RESHAPE`，去掉恒等的 `RESHAPE`/`PERMUTE`，合并 `PERMUTE` 链，把 `STACK(INDEX(b,0), INDEX(b,1), ..)` 折回 `b`，把 `INDEX(STACK(..), const)` 折叠为对应 lane，并在索引为标量时把 `INDEX(INDEX(b, i), j)` 组合为 `INDEX(b, i, j)`。这里不运行符号匹配器。

在完整示例中，reduce range `R2`（`Unroll`，extent 4）消失，索引变为带形状的：

```text
[151] REDUCE(Add, num_axes=1, ranges=[118]) : Scalar(Float32) shape=[]
├── [149] INDEX : Scalar(Float32) shape=[Const(4)]
│   ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
│   └── [148] Add : Scalar(WeakInt) shape=[Const(4)]
│       ├── [147] Add : Scalar(WeakInt) shape=[Const(4)]
│       │   ├── [119] Mul : Scalar(WeakInt) shape=[]          ← R0 * 4
│       │   └── [146] STACK(len=4) : Scalar(WeakInt) shape=[Const(4)]
│       └── [90] Mul : Scalar(WeakInt) shape=[]              ← R1 * 64
└── [118] RANGE(R0, Reduce)
```

`expand_reduce` 已经把 4 宽的 lane 轴变成了 `num_axes=1`，因此对 lane 的归约是水平的，剩下的循环只覆盖 `R0`。

:::tip[STACK 是唯一的向量算子]
带形状的值就是若干 lane 组成的 `STACK`（可能嵌套，可能位于 `RESHAPE` 之后）。`INDEX(STACK(..), c)` 用与缓冲区寻址相同的算子选出一个 lane。不存在 vectorize/contract 这样的算子对，`Upcast`/`Unroll` 是 `AxisType`，不是算子。
:::

## 10 —— 归约降级（`pm_reduce`）

`movement_cleanup_patterns() + pm_reduce_local()`，使用 `ReduceContext`。`movement_cleanup_patterns` 是 `mop_cleanup_patterns` 加上两条 devectorizer 专用规则（形状一致时 `RESHAPE(STACK([x]))` → `x`；只增加前导 1 维的 `RESHAPE` → 每个新增维度一个 `STACK([..])` 包装）。

`pm_reduce_local`（`devectorize.rs`）按顺序组合：

1. **`pm_wmma_add`** —— `WMMA(a, b, c) + add` → `WMMA(a, b, c + add)`，也能穿过 `expand_wmma` 留在输出上的 `PERMUTE` 和 `PERMUTE(RESHAPE(..))` 包装。`try_add` 在 dtype 不匹配时放弃，而不是断言失败。
2. **`pm_group_for_reduce`**（`expand.rs`）—— 带 `GroupReduce` range 的 `REDUCE` 变为：对其他 range 的部分 `REDUCE` → 用作用域内的 `Local` range 加上分组 range 对部分结果做 `STAGE`（`BufferizeOpts::local_for_axis`）→ 用这些 local range 和新的 `Reduce` 循环（`axis_id.group_reduce_loop()`）对该 stage 做 `INDEX` → 对这些循环做最终 `REDUCE`。
3. **`reduce_to_acc`** —— 带 range 的 `REDUCE`。若 `num_axes > 0`，先按行主序从左到右折叠各 lane（`horizontal_reduce`）。然后：

   ```text
   acc        = BUFFER(slot, AddrSpace::Reg)                       // placeholder_like(red)
   acc_init   = STORE(AFTER(acc, input_ranges), identity)           // 0 for Add, 1 for Mul, dtype min/max for Max/Min
   acc_loop   = AFTER(acc, [acc_init, reduce_ranges..])
   body       = op(acc_loop, horizontal_inp)                        // Add/Mul/Max; float Min is -(max(-a, -b))
   store_end  = END(STORE(acc, body), reduce_ranges)   tag=TAG_MERGEABLE
   result     = AFTER(acc, [store_end])
   ```

   `input_ranges` 是输入处作用域内既未被归约、也尚未被关闭的 range，因此初始化落在外层循环之内。这里没有循环结构：`END` 关闭 reduce range，`AFTER` 链就是数据依赖。
4. **`expand_horizontal_reduce`** —— 不再有 range 的 `REDUCE` 只剩下 lane 折叠。
5. **END 合并** —— 在 `SINK` 处，`merge_reduce_ends` 按 reduce range 集合和嵌套上下文对 `TAG_MERGEABLE` 的 `END` 分组，并把每组替换为 `END(GROUP(computations), ranges)`；处于不同嵌套深度的组会得到带新轴 id 的克隆 `RANGE`，使每个 range 恰好被一个 `END` 关闭。
6. **`clean_up_group_sink`** —— 单源 `GROUP` 被解包；`SINK` 或 `GROUP` 中的 `NOOP`/`STACK`/`SINK`/`GROUP` 源被展平。

浮点上的 `Min` 通过 `Max` 降级（`-(max(-a, -b))`），使 NaN 的行为与 max 归约一致；整数上则是 `WHERE(a < b, a, b)`。

## 11 —— local 缓冲区

`pm_add_local_buffers = { Stage => add_local_buffer } + movement_op_patterns`（`optimizer/mod.rs`）。存活到这里的每个 `STAGE` 都是 `pm_group_for_reduce` 刚刚创建的（全局的已在切分时变成 `STORE`，而 `bufferize_to_store` 有意跳过了 `Local` 的）。`add_local_buffer` 分配 `UOp::placeholder(max_shape, dtype, slot, opts.addrspace)` —— slot 是分组轴的 `LocalBufferContext::axis_slot`，对嵌套轴路径而言是确定性的哈希 —— 并把该 stage 重写为 `AFTER(buffer, [END(STORE(INDEX(buffer, ranges), compute), ranges)])`。随后 `movement_op_patterns` 把新 `INDEX` 之上的任何 movement 算子推入索引表达式。

Tinygrad 同样先降级归约再添加 local 缓冲区，原因相同：分组归约的 stage 要等 `pm_reduce_local` 的第 2 步运行之后才会存在。
