---
sidebar_label: 后期重写与线性化器
---

# 后期重写、程序边界与线性化器（阶段 19–20 及之后）

最后几个 post-optimization 阶段让图能够被某个具体后端渲染：目标缺少的操作被分解，有效性变成 `gate`，弱 dtype 被确定。然后 `svod-codegen` 添加控制流边、为参数编号，并把 DAG 展平为指令列表。源码：`optimizer/mod.rs`、`late/gater.rs`、`late/dtype.rs`、`optimizer/implicit_barriers.rs`、`linearize/`、`codegen/src/program_pipeline.rs`。

下面的每个匹配器都由 renderer 的能力表（`renderer.supported_ops()`、`supports_dtype`）构建，这也是 `optimize_kernel_with_config` 拒绝没有能力表的 renderer 的原因（`OptError::MissingRendererCapabilities`）。

## 19 —— 浮点 ALU 操作数类型转换

`pm_cast_float_alu`：对于 `Sin`、`Log2`、`Exp2`、`Sqrt`、`Reciprocal`，把操作数转换为结果 dtype。超越函数分解会展开为 dtype 同质的多项式，不能遇到 dtype 混杂的操作数。

## 19b —— 早期分解

`early_decomposition_patterns(supported_ops)`：

```text
symbolic_simple + pm_fold_cast_const + pm_mod_to_and + divmod_decomposition_patterns
  + pm_threefry_decomp        if !supports(Threefry)
  + pm_max_decomposition      if !supports(Max) && supports(Lt)
  + pm_erf_decomposition      if !supports(Erf)
```

`divmod_decomposition_patterns`（`ir/src/decompositions/mod.rs`）把向下取整除法和取模（`FloorDiv`/`FloorMod`）降级为带符号修正的截断式 `CDiv`/`CMod` —— 每个后端都具备的形式。`pm_mod_to_and` 既出现在这里也出现在后期规则集中，这样 2 的幂取模会在截断式降级看到它们之前被折叠。

## 19c —— dtype 分解

`pm_dtype_decomp_commit = pm_dtype_decomps + pm_commit_weak`，使用 `DTypeDecompCtx`。第一条规则只记录图中出现了 `FP8E4M3`、`FP8E4M3FNUZ`、`FP8E5M2`、`FP8E5M2FNUZ`、`Float16`、`BFloat16`、`Int64`/`UInt64` 中的哪些；随后 `SINK` 规则按 dtype 顺序自底向上地重写 renderer 不支持的每个已记录 dtype：

| 不支持的类型 | 模拟方式 | 匹配器 |
|-------------|-------------|---------|
| `Int64`、`UInt64` | 两个 `Int32`/`UInt32` 字：`PARAM`/`BUFFER` 大小加倍，`INDEX` 标记所用的字，进位和借位由 `Lt` 构建，`CDiv`/`CMod` 使用 64 步移位-减法除法器 | `pm_long_decomp`（`devectorize.rs`） |
| FP8 | 支持时用 `Float16`，否则用 `Float32`；存储仍为 8 位无符号字，由 `f2f` 完成逐位精确的转换（RNE 舍入、FNUZ NaN 编码、`f2f_clamp` 中的饱和处理） | `pm_float_decomp` |
| `Float16`、`BFloat16` | 用 `Float32` 计算，存储转换同样使用 `f2f` | `pm_float_decomp` |

`get_dtype_decomps` 以列表形式返回相同的选择，用作 renderer 编译缓存键的一部分。

## 19d —— 后期分解

`pm_decomp = early_decomposition_patterns + get_late_rewrite_patterns(renderer, disable_fast_idiv) + get_transcendental_patterns(supported_ops, TRANSCENDENTAL >= 2) (+ renderer.decomposition_matcher())`，运行到不动点。后期规则集受能力开关控制（所有规则见[强度削减](../optimizations/strength-reduction.md)）：

```text
pm_mod_to_and + pm_half_bf16_cast                       always
+ pm_demorgan                                           if supports(Or)
+ pm_mul_to_shl                                         if supports(Shl)
+ pm_div_to_shr                                         if supports(Shr)
  + fast_division_patterns + pm_mod_to_idiv             if supports(Shr) && DISABLE_FAST_IDIV=0
+ pm_neg_from_mul                                       if supports(Neg)
+ pm_comparison_negations                               if supports(Lt) || supports(Eq)
+ pm_fma_decomposition                                  if supports(MulAcc)
  + pm_shl_add_to_mulacc                                if supports(MulAcc) && supports(Shl)
+ pm_fdiv_to_mul                                        if supports(Fdiv)
```

`get_transcendental_patterns`（`ir/src/decompositions/`）对 renderer 缺少的每个算子（或在 `TRANSCENDENTAL=2` 时对全部算子），把 `Exp2`、`Log2`、`Sin` 替换为针对 f16/f32/f64 的 `xexp2`/`xlog2`/`xsin` 多项式近似（其他浮点类型在 f32 中计算），把 `Sqrt` 替换为 `xpow(x, 0.5)`。`decomposition_matcher` 是设备端 `Renderer::decompositor()` 钩子在优化器一侧的副本；Metal 在这里安装了 `amd_decomposition_patterns`（基于原生 `exp2`/`log2` 实现 `Exp`、`Log`、`Cos`、`Tan` 和二元 `Pow`）。

在完整示例中，这一阶段把 `R1 * 64` 变成 `R1 << 6`，把 `R0 * 4 + (R1 << 6)` 变成 `MulAcc(R0, 4, R1 << 6)` —— 由 `pm_shl_add_to_mulacc` 构建的整数 FMA。

## 19e —— gate、寄存器 lane、浮点降级

`pm_move_gates_from_index`（`late/gater.rs`，移植自 Tinygrad 的 `gater.py`）最终把有效性从索引中移出：

| 之前 | 之后 |
|--------|-------|
| `LOAD(INDEX(buf, WHERE(g, idx, Invalid)))`（无 `alt`，无 `gate`） | `LOAD { index: INDEX(buf, idx), alt: 0, gate: g }` |
| `STORE(INDEX(buf, WHERE(g, idx, Invalid)), v)`（无 `gate`） | `STORE { index: INDEX(buf, idx), value: v, gate: g }` |
| 作用在 `SHRINK` 上的同样两种形式（合并后的分组） | 作用在清理后 `SHRINK` 上的带 gate 的 `LOAD`/`STORE` |
| 共享同一条件的图像双坐标 `INDEX` | 一次带 gate 的访问（优先检查） |
| `WHERE(g, LOAD{gate: g}, alt)` 及其反向形式 | `alt` 折叠进 load |

`valid_index` 要求 `WHERE` 的第三个槽位是字面量 `Invalid` 常量。然后 `pm_scalarize_register_stack_index_preserve_deps` 把 `INDEX(AFTER(STACK(..), deps), c)` —— 在若干 store 之后读取寄存器 stack 的某个 lane —— 解析为被选中的那个 `LOAD`，并把 deps 重新挂到其地址上；`merge_register_read_ends` 把关闭相同 range 的 `END` 合并到同一个寄存器 `AFTER` 之下（一条调试断言检查没有寄存器 stack 的 `INDEX` 残留）。`demote_unsupported_floats`（`late/dtype.rs`）最后运行：在没有 `Float64` ALU 的 renderer 上（Metal、WebGPU），每个内部 f64 值都在 f32 中计算，而全局 f64 存储、对它的 load 及其 `alt` 值保持宽 dtype。

## 20 —— 最终重写

```text
pm_final = pm_commit_weak + pm_cast_weak + pm_decomp (+ renderer.extra_matcher()) + pm_split_ends
```

一次不动点迭代（调试构建中会先运行 `assert_target_renderer_boundary`：没有静态多索引 `INDEX`，没有残留的单例广播，没有 `STACK`/向量混合的 ALU）。`pm_split_ends` 把 `END(x, [r1, r2, r3])` 变成 `END(END(END(x, r3), r2), r1)`，range 按 `(axis_id, axis_type.priority())` 降序排列；`Void`/`Bool` 源（归约回边）被分离出来并重新挂到最外层 `END` 上，原始标签被保留，以便后续合并步骤仍能找到它。`extra_matcher` 是 `svod_device::device::Renderer` 上的按后端钩子；它与分解规则在同一个不动点中运行。CPU 和 NVPTX renderer 再次安装 `bool_storage_patterns`（`cpu_extra_matcher`），AMD 安装 `amd_non_native_fp8_patterns`（OCP FP8 ALU 加宽到 f32；存储、转换和 MFMA 操作数不受影响）。

然后是几个独立的 pass：`pm_remove_invalid` 把所有剩余的数据类型 `WHERE(c, x, Invalid)` 替换为 `WHERE(c, x, 0)`，把所有 `Invalid` 的 `STACK` lane 替换为零（一条调试断言检查没有残留）；`add_implicit_barriers` 为 local 内存插入 `BARRIER`：在 local 缓冲区上的 `AFTER` 之前、且其 deps 中含有未加屏障的 local `STORE` 时插入 RAW 屏障；在循环体末尾、若该循环向某个 local 缓冲区写入而同一循环中另有 load 读取它时插入 WAR 屏障。`optimize_kernel_with_config_and_final_rewrite` 返回插入屏障之前捕获的图，供一致性对比工具使用。被 `graph_rewrite` 丢弃的 `KernelInfo` 元数据会被重新挂上。

## 程序边界（`program_from_sink`）

由 `svod-codegen` 接手（`program_pipeline.rs`）：

1. **`add_control_flow`**（`linearize/mod.rs`）：再次运行 `pm_split_ends`（幂等），然后 `CFGContext::new(sink)`，并自底向上运行 `pm_add_control_flow`。该上下文为每个 `END` 计算它嵌套在哪个 `END`/`SINK` 之内 —— 当 `u` 依赖 `x` 且 `u` 的 range 在 `x` 的依赖中时，`END x` 嵌套在 `u` 内 —— 按父节点对兄弟节点分组，按它们依赖的兄弟数量排序，并记录一条从每个后续兄弟的 `RANGE` 指向其前驱的边（前一个兄弟的 `END`，第一个兄弟则指向父节点的 `RANGE`）。`pm_add_control_flow` 把前驱追加到该 `RANGE` 的源中；随后 `InScopeRangesProperty` 就能看到嵌套关系，这正是下文中嵌套 range 获得更大 `run_count` 的原因。若前驱本身已包含该 range，则 panic（`"edge would create cycle"`）。
2. **`number_params`** 分配最终的 `PARAM` slot（`validate_param_slots` 拒绝未分配或重复的 slot）。
3. 开启 `SVOD_SPEC` 时，针对 `spec_program` 执行 **`verify_final_sink`**；`ProgramInfo::from_sink` 读取 ABI。
4. **`pre_isel_matcher` / `isel_matcher`** —— `svod_device::device::Renderer` 上的两个指令选择钩子，都是自底向上的（`PreIselContext`、`IselContext`）。它们为 ISA 级后端而设；LLVM 和 C renderer 将其留为 `None`。
5. `UOp::program(sink, info, None, None, None)` —— `PROGRAM` 节点，其后续的源是 `LINEAR`、`SOURCE` 和 `ProgramBinary` 阶段。

## `linearize`

`do_linearize` 调用 `svod_schedule::linearize(sink)`（`linearize/linearize.rs`，直接移植自 Tinygrad 的 `linearizer.py`），然后是 `line_rewrite_cleanups`，再针对 `spec_program` 执行 `verify_linear_list`。

每个节点的排序键是 `(run_count, priority, extra, tuplize rank)`：

| 算子 | 优先级 |
|----|----------|
| `PARAM` | −20，平局按 slot 决定（`extra`） |
| `BUFFER`（全局、寄存器） | −18 |
| `BUFFER`（`AddrSpace::Local`） | −17 |
| `END` | −5 |
| `LOAD` | −1 |
| 其余一切（`CONST`、ALU、`SPECIAL`……） | 0 |
| `STORE` | +1 |
| `RANGE` | +5 |

`run_count = prod(vmax + 1)`，在节点作用域内的 range 上求积（符号 extent 计为 1），因此循环之外的代码排在循环体之前。tuplize rank 即 Tinygrad 的 `(op, arg, dtype, *src.tuplize)` 键，在拓扑序上迭代计算（`TuplizeKeys`），使排序成为全序且确定。然后从 `SINK` 出发、基于这些排名做最大堆拓扑排序，最后反转，得到线性列表：一个节点在其所有消费者都已输出后才输出，因此定义在前、`LOAD` 在其使用之前、`STORE` 在计算之后，每个 `RANGE` 紧挨着其循环体打开。

`SVOD_DUMP_LINEAR=<dir>` 写出带作用域内 range id 的拓扑序（`tree_<id>.txt`）和最终列表（`linear_<id>.txt`）。

## `line_rewrite_cleanups`

`line_rewrite` 遍历一次指令列表；每个条目可以被替换为多个条目。唯一的清理是 `linearize_cleanup_pattern`：若一个 `STORE` 带有 `Bool` gate 且其地址是 `INDEX`/`SHRINK`（可能位于 `CAST` 之后），它会变成 `IF(gate)`、不带 gate 的 `STORE`、`ENDIF`。`IF` 和 `ENDIF` 都只存在于列表中，从不出现在图中（`"if not allowed in graph"`），这就是为什么 `spec_program` 既在 sink 上检查、也在列表上检查。能对 store 做谓词化的后端（LLVM、CUDA、Metal）把这个三元组渲染为条件 store。

该列表就是 `PROGRAM` 的 `LINEAR` 阶段；`do_render` 把它交给 `Renderer::render`，`do_compile` 生成二进制。[完整示例](./worked-example.md)展示了 CPU renderer 为行求和内核生成的 LLVM IR。
