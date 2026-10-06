---
sidebar_label: Range 与归约
---

# Range 与归约优化

这些规则决定哪些循环存在：拆分、合并和收窄 range，把归约折叠为闭式解，内联或物化中间结果。它们位于 `schedule/src/rangeify/{patterns,transforms,kernel}.rs`，在 rangeify 的 mega-pass、内核切分以及 `apply_pre_optimization` 中运行（顺序见 [Rangeify](../codegen/rangeify.md)）。Tinygrad：`schedule/rangeify.py`、`codegen/simplify.py`。

## Range 拆分（`pm_split_ranges`）

满足 `end % c == 0` 的 `RANGE % c` 会标记该 range；在 `SINK` 处，每个被标记的 range 被替换为 `outer * c + inner`，其中 `outer = RANGE(end / c)`、`inner = RANGE(c)`，两者都保留原轴类型，轴 id 分别为 `axis.child(0)` / `axis.child(1)`（不分配全局 id）。`Warp` 和 `Device` range 从不拆分；图像 `STORE` 索引到的每个 range 都被固定，因为图像地址是坐标对，而不是扁平偏移。替换后的图用 `symbolic + pm_fold_cast_const` 化简，因此 `inner % c → inner` 和 `(outer*c + inner) // c → outer` 会立即触发。

## Range 合并与收窄（`pm_simplify_ranges`）

`simplify_merge_adjacent` 在每个至少有两个 range 的 `END` 和 `REDUCE` 上运行。对 `END` 它尝试相邻的对；对 `REDUCE` 则尝试每个有序对。当一对 `(r0, r1)` 轴类型相同、end 为常量、且出现在相同的 `REDUCE` 中（作用域一致）时，它们会被合并：合并后的 range `R(s0*s1)` 用 `R // s1` 替换 `r0`、用 `R % s1` 替换 `r1`，图用 `symbolic + pm_fold_cast_const + pm_flatten_range` 化简，并且只有当 `FloorDiv`/`FloorMod` 的数量没有增加时才保留合并（`count_divmod`，逐节点记忆化）。符号 end 从不合并：divmod 数量不会改变，而符号乘积会把常量轴对之后所有只接受常量的优化（upcast、unroll、locals、tensor core）隐藏起来。

`mark_gated` 从每个 `INDEX` 收集每条有效性子句 `range < c` 为某个 range 证明的界限；只要某个 range 在任何地方有一处无 guard 的使用，就被固定为其自身的 end，并且 `REDUCE` 的 range 受保护。在 `SINK` 处，每个有界的 range 用已证明的最大界限重建，并化简结果。连同 `pm_flatten_range`（由经源可达的 `RANGE` 重新推导 range 列表，保留 `Bool`/`Void` 回边），这就是“化简 range”阶段的全部内容。

## Load collapse（`pm_load_collapse`）

`reduce_load_collapse(src, ranges)`，对每个 range：取该 range 作用域内的节点（遇到嵌套的 `REDUCE` 或 `STORE` 则放弃），把每个不是常量或 `PARAM` 的外部输入替换为携带其 `vmin`/`vmax` 的标量 `PARAM` 变量 `in{n}`（`UOp::variable`），把主体包进覆盖该 range 的合成 `REDUCE(Add)`，然后运行 `build_reduce_load_collapse_matcher`。若没有 `RANGE` 幸存，则把变量替换回去。该匹配器是 `pm_reduce_collapse` 加上 `.or_casted()` 形式以及 `NE` 提升。

界限规则（`reduce_collapse_inner_patterns`，Tinygrad `simplify.py`）：

| 归约主体（在 `r ∈ [0, N)` 上） | 闭式解 |
|---------------------------------|-------------|
| `WHERE(r < cut, 0, v)` | `clamp(N - cut, 0, N) * v` |
| `WHERE(r < cut, v, 0)` | `clamp(cut, 0, N) * v` |
| `WHERE(r >= lo & r < hi, v, 0)` | 双侧 clamp 乘以 `v` |
| `WHERE(idx != r, 0, e)`、`WHERE(idx == r, e, 0)`（gather） | `WHERE(0 <= idx < N, e[r := idx], 0)` |

（clamp 内部的 `min` 写作 `-max(-a, -b)`，以便 `Max` 界限规则能处理边界情况。）围绕它们的还有：`pm_reduce_unparented`；暴露界限的提升变换 —— `(x + y) < c → x < c - y` 和 `(x*y) < c → x < ceil(c/y)`，也能穿过 `CAST`，`>=` 和 `==` 同理，load-collapse 变体中还包括 `!=`；分配律 `sum(x + y) → sum(x) + sum(y)`；`x * bool.cast() → WHERE(bool, x, 0)`；对于“与 range 无关的 `PARAM` 子句 AND 一个 range 子句”这种条件，使用 `try_param_factor`。外层的 `pm_load_collapse` 还会在 `x` 含有 load 时撤销已提升的 `(x + y) < c`，使被加载的索引永不溢出。使用更窄匹配器（没有 `!=` 提升）的同一引擎是 `reduce_collapse`，由 mega-pass 中的 `pm_reduce_simplify` 用于 `num_axes == 0` 的 `REDUCE(Add)`。

```text
sum(1 for k in 0..64 if k >= length)   →   max(0, 64 - length)
```

## 未引用的归约与因子外提（`pm_reduce_simplify`）

`pm_reduce_unparented`：主体未引用的 reduce range 被移除 —— `Add` 把结果乘以 extent，`Mul` 把结果取 extent 次幂，`Max` 直接丢弃该 range；`Min` 不会被匹配。`reduce_mul_chain`：在 `REDUCE(a * b * .., Add | Max)` 中，不依赖任何 reduce range 的因子被移到外面（对 `Max` 只移出可证明非负的因子），仅限整数。两者也都在 `POST_OPT_SYM`（阶段 08）和 `sym` 层级中运行。

## 缓冲区移除（`pm_remove_bufferize`）

`INDEX(STAGE(src, ranges, opts), indices)` 通过用消费者的索引替换 stage 的 range 来内联（`substitute_gated`；跳过 `CONST` range 和 `Invalid` 索引），除非：

1. `src` 是总是运行的算子（`CONTIGUOUS`、`COPY`、`NOOP`），或者该 stage 不可移除（`COPY` 的消费者、总是连续的源、多消费者的 realize 边界、自定义内核的输入）；
2. 计算读取了三个以上不同的缓冲区（`AFTER` 缓冲区、全局 `STAGE`、`MSTACK`、`PARAM`/`BUFFER`），这会让内核的参数列表膨胀；
3. 计算内部的某个 `REDUCE` 读取了缓冲区（`PARAM`、`BUFFER` 或 `STAGE`）—— 内联会在每次迭代中重新执行读取（`argmax(-x)` 会加载 `x` N 次而不是一次）。对不涉及任何缓冲区的值做归约仍然可以内联。

替换之后有两条清理规则：`STORE(x, x)` → `NOOP`，`END(NOOP)` → `NOOP`。

`buffer_folding`：`STAGE(CONST)`、`INDEX(CONST)`、`COPY(CONST)` 和 `INDEX(MSTACK(CONST, ..))` 折叠为该常量；range 相同的 `INDEX(STAGE(compute, ranges), ranges)` 即为收缩到 stage 形状的 `compute`，并合并标签。

`dead_axis_removal`：可移除的 `STAGE`（不位于 `AFTER` 或总是运行的算子之上，没有符号 end）丢弃为 `CONST` 或计算中未使用的 range，然后用 `RESHAPE` 补回大小为 1 的维度，再 `EXPAND` 到原始形状。一个 stage 可以最终没有任何 range；但它必须仍然存在，否则切分时不会产生 `STORE`。

## 两阶段归约（`split_reduceop`）

在最早期重写中，输入/输出比达到 `SplitReduceOpConfig::split_threshold`（32768）的张量形式 `REDUCE` 会被拆分：若某个被归约维度没有被广播（`detect_expanded_dimensions`），且能被 `[8, 256]` 中的某个因子整除（从最大的开始），并使中间输出保持在 `2^22` 个元素以下，则它被 reshape 为 `[.., divisor, rest, ..]`，在原始轴上归约，用 `CONTIGUOUS` 物化，再在 divisor 轴上归约一次。第一阶段于是有 `divisor` 个输出可供并行；第二阶段很小。

## 分组归约

这不是 rangeify 规则，但属于同一家族：`GROUP`/`GROUPTOP` 优化把 `Reduce` 轴的一部分变成 `GroupReduce`，`pm_group_for_reduce`（阶段 10）将其降级为一个暂存在 local 内存中的部分 `REDUCE`，再用新的 `Reduce` 循环（`axis_id.group_reduce_loop()`）读回并再次归约。参见 [Expander 页面](../codegen/expander.md)。
