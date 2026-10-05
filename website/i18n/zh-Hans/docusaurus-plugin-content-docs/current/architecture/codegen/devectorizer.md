---
sidebar_label: Devectorizer 与索引降级
---

# GPU 维度、去向量化与索引降级（阶段 12–18）

归约降级之后，内核仍然使用带形状的值和 `WeakInt` 索引。这些阶段把 range 映射到硬件索引，让每次内存访问都成为显式的标量 `LOAD`/`STORE`，把连续访问重新加宽，并确定索引 dtype。源码：`gpudims.rs`、`devectorize.rs`、`late/coalesce.rs`、`symbolic/index_lowering.rs`。

## 12 —— GPU 维度

两个匹配器。`pm_lower_device_ranges` 对每个 renderer 都运行：`Device` range 变成带该 range 界限的标量 `PARAM` 变量 `_device_num`，关闭它的 `END` 去掉对应条目。`pm_add_gpudims` 仅在 `renderer.has_local || renderer.has_threads` 时运行；它只匹配一次 `SINK`（`GpuDimsContext` 记住已降级 sink 的 id，使引擎的再次访问成为空操作）。

`add_gpudims`：

1. 收集每个 `RANGE`，以 `(axis_id, axis_type)` 为键；如果已存在 `SPECIAL` 则放弃。
2. 全局维度 = `Global` 和 `Thread` 轴；局部维度 = `Local`、`Warp`、`GroupReduce`。两者都按轴 id 排序；`Warp` 轴被移到局部维度的最前面，使其占据线性线程索引的低位（`mma.sync` 按硬件 lane 寻址 fragment）。
3. 构建索引表达式：
   - `has_threads`（CPU）：必须恰好一个全局轴且没有局部轴，否则该 pass 发出警告并放弃。该轴变成 `PARAM("core_id", 0..N-1)`。
   - `KernelInfo.dont_use_locals`：只有全局维度，`get_grouped_dims("idx", ..)`。
   - 其他情况：先在 `local_max_axes()`（或逐轴的 `local_max`；首个上限被固定为 warp 的 extent，使其他维度不会折叠进 `lidx0`）约束下由局部形状生成 `lidx*`，然后在 `global_max` 约束下由全局形状生成 `gidx*`；当 renderer 声明了工作项乘积上限时，再进一步以 `global_prod_max / hardware_local_extents` 为上限。
4. `get_grouped_dims` 与 Tinygrad 相同：若维度超出逐轴上限，`group_dims` 合并乘积不超限的相邻维度；若无法分组，`split_dims` 用最小因子拆分过大的维度，放入下一个槽位。任一失败都会在调度时 panic，而不是在代码生成时（`"cannot limit dims to N axes"`）。结果是每个受限维度一个 `SPECIAL(end, "gidxN")`；发生分组或拆分时，每个原始维度通过 `FloorDiv`/`FloorMod` 从扁平索引重建，并用 `symbolic` 化简。全局索引以 `reverse = true` 生成（递归会同时反转输入*和*输出，因此名称保持迭代顺序）。
5. **Store 掩码**（`compute_store_masks`）：若写入全局内存的 `STORE` 的索引不在每个局部 range 的作用域内，则其索引会得到 `WHERE((l1 == 0) & (l2 == 0) & .., idx, Invalid)`，使每个未使用的局部轴只有一个工作项执行写入。掩码留在索引表达式内部，这样 RANGE → SPECIAL 的替换会把它带到硬件索引上。
6. 把每个 GPU range 替换为其索引；`Reduce` range 仍为循环。

在[完整示例](./worked-example.md)页面的 CPU 例子中，这里什么也不发生：行轴是 `Weak`（没有创建 `Thread` 轴），并且该 renderer 没有局部维度。

## 13 —— loads

`PM_ADD_LOADS = symbolic_simple() + pm_expand_broadcast() + pm_add_loads()`。

- `pm_expand_broadcast` 先再次运行 `pm_wmma_add`，然后把广播显式化：若 `Binary`/`Ternary`/`STORE` 的源形状不同，每个源都被 `RESHAPE`（前导 1）并 `EXPAND` 到广播形状；操作数前缀不同的 `WMMA` 按输出坐标展开（`broadcast_and_devec_wmma`）。
- `pm_add_loads` 把每个*作为值被消费*的操作数包进 `LOAD`：带地址空间的 ALU 算子、类型转换、`REDUCE`、`WMMA` 和 `STACK` 的源（`maybe_load`），以及本身是地址的 `STORE` 值。用作地址的 `INDEX` —— `STORE` 的目标、`WMMA` 的 fragment 指针 —— 保持原样。阶段 10 创建的累加器读取（`AFTER(acc, ..)`）在这里变成 `LOAD(AFTER(acc, ..))`。

## 14 —— 去向量化

`devectorize()` 是基于 `symbolic_simple + devectorize_patterns + bool_storage_patterns + indexing_simplify` 的一次 `graph_rewrite`（`Renderer` 上下文，规则并不使用它）。没有外层循环：引擎会对每个替换结果重新匹配。

`devectorize_patterns`（Tinygrad 中的 `devectorizer2`），按源码顺序：

| 组 | 规则 |
|-------|-------|
| `movement_cleanup_patterns` | `mop_cleanup_patterns`，加上 `RESHAPE(STACK([x]))` 和前导单例维度的物化 |
| `movement_op_patterns` | rangeify 的 movement 规则（穿过 `INDEX`、`AFTER`、`END`） |
| `no_vectorized_alu` | 形状非空的每个一元/二元/三元算子、`CAST`、`BITCAST` → `devectorize_alu` |
| `mixed_representation_alu` | 源中混有 `STACK` 和向量 dtype 值的 ALU：向量源被拆成 `STACK(INDEX(src, lane)..)`，然后 `devectorize_alu` |
| 带形状的 `LOAD` / `STORE` | → `devectorize_alu`（逐 lane 的 `LOAD(INDEX)`；逐 lane 的 store 收集到一个 `GROUP` 中） |
| `INDEX(buf, [])` | → `buf` |
| `WMMA` | `stack_wmma_sources`：操作数变成已加载 lane 的 `STACK` |
| `PARAM`/`BUFFER` 上的 `INDEX(buf, STACK(i0, i1, ..))` | → `STACK(INDEX(buf, i0), INDEX(buf, i1), ..)` —— lane 仍是地址；外围的 `LOAD`/`STORE` 负责将其物化 |
| `INDEX(buf, RESHAPE(i))` | → `RESHAPE(INDEX(buf, i))` |
| `Void` 值的 `RESHAPE` | → 该值（围绕 `AFTER`/`STORE` 的形状簿记） |
| 被 reshape 为标量的单元素带形状值 | → `INDEX(src, 0)` |
| `EXPAND` | `materialize_stack_broadcast`（广播到 N 个 lane 的 `STACK([x])` → `STACK([x; N])`）或 `expand_scalar_to_stack` |

`devectorize_alu` 即 Tinygrad 的 `do_devectorize`：它要求每个源都具有结果形状（或者是 `Invalid` 基值，其标量是多态的），枚举静态形状的所有坐标，为每个坐标构建一个以 `INDEX(source, c0, c1, ..)` 为操作数的标量算子，再用 `stack_with_shape`（与形状对应的嵌套 `STACK`）重新组装 —— 对 `STORE` 则用 `GROUP`。lane 数是形状的完整乘积；没有按设备区分的折叠宽度。重新向量化是后端的工作（LLVM 的 SLP 向量化器，或对内存访问而言是两个阶段之后的 `memory_coalescing`）。

`bool_storage_patterns`：bool 的 `STORE` 转换为 `uint8`，bool 的 `LOAD` 加载 `uint8` 后再转换回来，涉及 bool 的 `BITCAST` 变成 `CAST`。LLVM 的 `i1` 高位可能带有垃圾值。

`indexing_simplify`（`late/coalesce.rs`）：对于 `INDEX(buf, WHERE(valid, idx, Invalid))`，`uop_given_valid` 在 `valid` 成立的假设下重写 `idx`（`symbolic/valid_simplification.rs`）；双坐标的图像形式还会丢弃图像边界已经蕴含的有效性子句（`drop_valid_stmts`）。

此阶段之后，每个 ALU 算子都是标量的。在完整示例中，索引表达式的四个 lane 变成四个 `LOAD(INDEX(PARAM, R0*4 + R1*64 + k))`，水平 `Add` 链也变得显式。

## 15 —— 早期符号化简

再运行一次 `sym()`，这次是在标量代码上。它存在的理由在于下一阶段：索引表达式必须处于规范的 `base + const` 形式，合并访存才能对其分组。

## 16 —— 内存合并访问

`memory_coalescing`（`late/coalesce.rs`）是图遍历，不是匹配器。它按 `(op, buffer, index base, validity)` 对无 gate 的 `LOAD` 和 `STORE` 分组，其中索引被拆成 `base + integer_offset`（`Invalid` 或常量索引自成一个 base）。组内连续的偏移构成若干段；每段被切成能整除 base 偏移的最宽折叠长度：

- 图像缓冲区：4；
- `supports_float4` 的 renderer：当 `access_bytes() >= 16` 时从 `16 / sizeof(dtype)` 起向下取 2 的幂（8 个 16 位 lane、4 个 `f32`），否则从 4 起；
- 其他情况：只能是标量。`Reg` 缓冲区和不可折叠的 dtype（f32/f16/bf16/i32/u32/fp8 以外的一切）保持标量。

宽度为 `n > 1` 的折叠变成 `LOAD(SHRINK(buf, offset, n))`，旧的 load 被替换为 `INDEX(load, lane)`；或变成 `STORE(SHRINK(..), STACK(values))`。`SHRINK` 携带分组形状；内存 dtype 仍为标量。`DMC=1` 禁用此 pass。在完整示例中，四个展开的 load 变成一个 `LOAD(SHRINK(PARAM(1), R0*4 + R1*64, 4))`，LLVM 后端将其渲染为 `load <4 x float>`。

## 17 —— 自底向上的逐元素 / 图像 pass

`symbolic_simple + no_vectorized_alu + pm_simplify_add_image`，通过 `graph_rewrite_bottom_up` 和 `AddImageContext` 应用。图像规则把对 f32 图像缓冲区的 f16 访问规范化（`LOAD` → `LOAD.cast(f16)`、`STORE(value.cast(f32))`，并去掉 `CAST(CAST(x, f16), f32)` 往返转换）。图像缓冲区的*创建*在 Svod 中没有对应目标；这些规则只服务于已有的图像访问。`no_vectorized_alu` 再次运行，因为图像重写可能重新引入带形状的算子。

## 16 —— 额外符号化简

`extra_symbolic_patterns = sym() + indexing_simplify()`。这里的索引有意仍为 `WeakInt`：`sym` 和 `indexing_simplify` 的分配律规则与索引有效性规则需要弱 dtype，所以这是它们最后的机会。（该标签与内存合并访问的标签冲突；两者都在 `SVOD_DUMP_STAGE=16` 下打印。）

## 17 —— 索引 dtype 降级

`lower_index_patterns = symbolic_simple + pm_fold_cast_const + pm_lower_index_dtype + indexing_simplify`，每个内核一个 `WeakMemo`（对应 Tinygrad 的单个 `ctx={}`）。移植自 `tinygrad/uop/weak.py`。

`select_dtype(u)`：`WeakFloat` → 默认浮点类型；`vmin`/`vmax` 落在 `i32` 范围内的整数 → 默认整数类型，否则为 `Int64`；向量数量保持不变。

`pm_lower_index_dtype` 组合了：

1. `pm_commit_weak` —— 若 `Binary`/`Ternary` 带有弱类型源且 `least_upper_dtype` 非弱，则把弱类型源确定为该类型（`commit_weak`：`CONST` 直接改类型，其他一律做类型转换）；值为弱类型的 `STORE` 把值确定为索引的 dtype。
2. `pm_cast_weak` —— `CAST(weak_alu, concrete)` 把具体 dtype 推入该 ALU 的源。
3. `SHRINK` 的偏移/大小用 `select_dtype` 确定。
4. 任何带弱类型源的非弱节点 → `lower_weak_srcs`：每个弱类型源用 `pm_lower_weak` 重写（按源 id 记忆化），末尾的弱 `CAST` 由消费者自己的边吸收。`pm_lower_weak` 是三阶段级联：
   - 叶子：`CONST`/`VCONST`/标量 `PARAM` 变成 `concrete.cast(weak)`；
   - `Unary`、`Binary`、`WHERE`（跳过条件）、`RANGE`、`STACK`、`SPECIAL`（`lower_weak_node`）：解开源上的弱类型转换，计算具体 dtype（二元算子取 `select_dtype(u)` 与各源的 `least_upper_dtype`；其他情况用 `dtype_from_op`），把每个源转换为该类型，除 `STACK` 外在结果上保留一个弱 `CAST`；
   - 弱 dtype 的 `INDEX`：缓冲区转换为选定的 dtype，每个弱索引被确定；
   - `CAST(weak, CAST(weak, x))`：确定内层转换，保留外层。
5. 若 `INDEX`（或 `SHRINK`）的带 gate 索引已经是 `Int64`，当缓冲区元素数落在 `i32` 范围内时会被收窄回 `Int32`。

在这一阶段 `WHERE(valid, idx, Invalid)` 保持其形状：`lower_weak_node` 不会触碰 `Invalid` 源。在完整示例中，每个 `WeakInt` 都变成 `Int32`（`RANGE(R0, Reduce) : Scalar(Int32)`），`PARAM` 的大小也变成 `Int32` 常量。

## 18 —— 最终符号化简

在具体类型的图上运行 `symbolic()`（第 2 层，没有 `pm_simplify_valid`/lane 折叠）。开启 `SVOD_SPEC` 时，`verify_no_legacy_index_dtype` 断言没有 `WeakInt` 残留。
