---
sidebar_label: 代数化简
---

# 代数化简

符号化简器是 `schedule/src/symbolic/patterns.rs` 中三个嵌套的匹配器。各层级在何处运行：

| 匹配器 | 组成 | 运行位置 |
|---------|-------------|---------|
| `symbolic_simple()` | `symbolic_simple_base() + dead_loop_patterns()` | add-loads（13）、devectorize（14）、图像 pass（17）、索引降级（17）、早期分解（19b），以及最终重写中的 `pm_decomp` 规则集 |
| `symbolic()` | `symbolic_simple` + 第 2 层各组 | rangeify mega-pass、range 拆分/合并（`+ pm_fold_cast_const`）、`indexing_simplify`、最终符号化简（18） |
| `sym()` | `symbolic` + 第 3 层 | 预优化（`+ pm_fold_cast_const + pm_flatten_range`）、post-opt 符号化简（08）、早期符号化简（15）、额外符号化简（16，`+ indexing_simplify`） |

`pm_fold_cast_const`（`CAST(CONST) → CONST`）有意*不*放在任何层级中；需要它的位置显式添加，正如 Tinygrad 只在 `UOp.simplify` 所在之处组合 `symbolic + pm_fold_cast_const`。

每条重写整数运算的规则都经过 `exact_integer_rewrite`，这是一个带类型的无回绕证明（`typed_integer_rewrite_is_exact`）：只要原表达式或替换表达式可能溢出其具体 dtype，就放弃该重写。标记为*值敏感*的组还额外被 `value_sensitive` 包装，在子树满足 `weak_float_values_are_committed` 之前禁用它们。界限来自 `VminVmaxProperty`（总是可用）和 `SoundVminVmaxProperty`（对界限不可信的算子返回 `None`：load、`Pow`、`Fdiv`），两者都按节点缓存。

## 组成

```text
symbolic_simple_base()         tier 1
  propagate_invalid                     must be first (before x*0 → 0)
  fold_invalid_load_store
  constant_folding_dsl_patterns         value-sensitive
  vconst_folding_patterns               value-sensitive
  bool_arithmetic_patterns
  identity_and_zero_patterns            value-sensitive
  self_folding_dsl_patterns
  zero_folding_dsl_patterns
  division_dsl_patterns                 value-sensitive
  cast_dsl_patterns
  uint_pack_dsl_patterns
  div_mod_recombine_dsl_patterns
  power_dsl_patterns                    value-sensitive
  boolean_dsl_simple_patterns
  dce_dsl_simple_patterns               value-sensitive
symbolic_simple() = base + dead_loop_patterns

symbolic() = symbolic_simple + tier 2 (with_tier2)
  commutative_canonicalization
  boolean_dsl_patterns
  term_combining_dsl_patterns           value-sensitive
  dce_dsl_patterns
  where_alu_combining_patterns
  vmin_vmax_collapse_patterns           value-sensitive
  minmax_dsl_patterns                   value-sensitive
  alu_folding_dsl_patterns              value-sensitive
  comparison_dsl_patterns               value-sensitive
  range_based_mod_div_patterns
  advanced_division_dsl_patterns
  range_based_cast_patterns
  long_to_int_narrowing_patterns
  after_simplification_patterns
  where_bound_patterns                  value-sensitive

sym() = symbolic + tier 3
  pm_simplify_valid                     (symbolic/valid_simplification.rs)
  alu_vectorize_reorder_patterns
  ne_zero_fold_patterns                 value-sensitive
  cast_where_dsl_patterns
  store_load_folding_patterns
  reduce_sym_patterns                   value-sensitive
  sym_phase3_patterns
```

顺序至关重要：规范化在项合并之前，ALU 折叠在比较规则和 range 规则之前，因为每一组都会为下一组暴露出匹配机会。Tinygrad 的倒数分配规则（`uop/symbolic.py` 中的 `sym`）被有意省略：六条全都不满足 IEEE 精确性。

**记号。** `OP[a, b]` 表示可交换，`OP(a, b)` 表示有序；`@zero`/`@one`/`c` 是常量；重复的名称表示同一节点（`Arc::ptr_eq`）。`//` 是 `FloorDiv`，`%` 是 `FloorMod`；截断式的 `CDiv`/`CMod` 只在后期分解之后出现。

## 第 1 层

### Invalid 传播（`propagate_invalid`）

`Invalid` 即 `UOp::invalid_marker()`，一个 `ConstValue::Invalid` 常量；`is_invalid_marker` 也能识别全为 `Invalid` 的 `VCONST` 或 `STACK`，以及包裹它们的 movement 外壳。有效性以 `WHERE(cond, x, Invalid)` 的形式保存，这些规则在算术运算围绕它移动时保持该形状不变：

| 模式 | 结果 |
|---------|--------|
| `WHERE(Invalid, _, _)` | `Invalid` |
| `WHERE(WHERE(c, x, Inv), a, b)` | `WHERE(c, WHERE(x, a, b), Inv)` |
| `WHERE(c, Inv, x)` | `WHERE(!c, x, Inv)`（`WHERE(c, Inv, Inv)` → `Inv`） |
| `WHERE(c1, WHERE(c2, x, d), d)` | `WHERE(c1 & c2, x, d)` |
| `WHERE(a, WHERE(c, x, Inv), y)`，`y` 不是 `Invalid` | `WHERE(!a \| c, WHERE(a, x, y), Inv)` —— 假分支有对称的规则 |
| `unary(Inv)`、`CAST(Inv)`、`BITCAST(Inv)` | `Inv` |
| `unary(WHERE(c, x, Inv))`、`CAST`、`BITCAST` | `WHERE(c, unary(x), Inv)` |
| 对**每个**二元算子（包括比较）的 `op(WHERE(c, x, Inv), y)`、`op(y, WHERE(c, x, Inv))` | `WHERE(c, op(x, y), Inv)` |
| 对 13 个非比较二元算子的 `op(Inv, y)`、`op(y, Inv)` | `Inv` |

为什么放在最前：`MUL(0, WHERE(c, x, Inv))` 必须变成 `WHERE(c, 0, Inv)`，而不是 `0`。

### 死 load 与死 store（`fold_invalid_load_store`）

`LOAD(INDEX(buf, Invalid, ..))`（也包括位于 `CAST` 之后的情形）→ 若 load 有 `alt` 则取之，否则为带形状的零；不带 gate 的 `STORE(INDEX(buf, Invalid, ..), v)` → `NOOP`。

### 常量折叠

一元：`Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc`（这里 `Neg` 不是算子 —— `neg()` 构建的是 `MUL(x, -1)`）。二元：`Add, Mul, Sub, FloorMod, Max, Pow, FloorDiv, Fdiv, And, Or, Xor, Shl, Shr`，外加结果为 `Bool` 的六种比较。三元：`Where`、`MulAcc`。结果按 dtype 的存储格式确定（`Int32` 会回绕），弱 dtype 除外，它们保留未截断的值。`vconst_folding_patterns` 对 `VCONST ⊕ VCONST` 以及 `CONST`/`VCONST` 广播混合逐 lane 做同样的事（11 个二元算子 + 比较，6 个一元算子），跳过弱类型 lane。

### Bool 运算

当两者都是 `Bool` 时，`Mul[x, y]` → `x & y`，`Add[x, y]` → `x | y`，`Max(x, y)` → `x | y`。

### 恒等与零

| 模式 | 结果 | 条件 |
|---------|--------|-------|
| `Add[x, 0]` | `x` | 非浮点，或该零是 `-0.0`（对 `x = -0.0`，`x + 0.0` 不是恒等变换） |
| `Sub(x, 0)` | `x` | 非浮点，或该零是 `+0.0` |
| `Mul[x, 1]`、`Or[x, 0]`、`Xor[x, 0]`、`FloorDiv(x, 1)`、`Fdiv(x, 1)` | `x` | |
| `FloorMod(x, 1)` | `0` | |
| `Floor/Ceil/Trunc/Round(x)` | `x` | 整数 `x` |
| `Mul[x, 0]` | `0` | 非浮点（`NaN * 0`、`Inf * 0` 为 `NaN`） |
| `And[_, 0]` | `0` | |

### 自身折叠与零折叠

`FloorDiv(x, x)` → `1`；`FloorDiv(x, -1)` → `MUL(x, -1)`；`FloorMod(FloorMod(x, y), y)` → `FloorMod(x, y)`；`And(x, x)`、`Or(x, x)`、`Max(x, x)` → `x`；`FloorMod(x, x)` → `0`；`Lt(x, x)` → `false`，对非浮点成立，对浮点则需可靠界限证明 `x` 不是 `NaN`；`Ne(x, x)` → `false`，对整数和 bool 成立。

### 除法

`Fdiv(0.0, 0.0)` 和 `Fdiv(MUL[_, 0.0], 0.0)` → `NaN`（列在最前，以便优先于下一条规则）；`Fdiv(x, x)` → `1.0` 仅当 `x` 可证明为有限且非零；`FloorDiv(Mul(x, y), y)` → `x`。没有浮点版本的 `(x*y)/y → x`。

### 类型转换

dtype 已经匹配时 `CAST(x, dt)` → `x`；当 `x: b` 且 `can_safe_cast(b, a)`（`a` 能容纳 `b` 的每个值：符号性相同且至少同样宽，无符号→有符号需要多一位，浮点↔整数永远不行）时，`CAST(CAST(x, a), b)` → `x`；当 `a` 不会收窄 `x` 时，`CAST(CAST(x, a), b)` → `CAST(x, b)`。`uint_pack_dsl_patterns` 消除 Threefry 构建的 `(hi.cast(u64) << 32) | lo.cast(u64)` 打包，使 PRNG 保持在 32 位 ALU 中。

### 除法与取模的重组

作用于每个 `Add` 的一条规则：`fold_add_divmod_recombine`，移植自 Tinygrad。它展平 `Add` 链，寻找一个项 `(base % div) * mul`，以及一个伙伴项 `q * (div * mul)`，其中 `q` 是某个模 `div` 与 `base` 同余之量的商（`quotient_base`：`q == b // div`，可能带有合并后的 `(x//c + a)//div` 和平移常量），并把这对项替换为 `b * mul`；若 `q == (b // div) % d`，则折叠为更宽的 `(b % (div*d)) * mul`。这就是 `x%n + (x//n)*n → x` 这一族规则及其缩放、偏移和三项变体，通过遍历链来发现，而不是写成单独的规则。

### 幂、布尔与 DCE

`Pow(x, 0)` → `1`、`Pow(x, 1)` → `x`、`Pow(1, x)` → `1`（仅限标量；其他指数都不重写 —— 倒数/平方根形式会改变 IEEE 舍入）。`Not(Not(x))` → `x`、`Xor(x, x)` → `0`、`true | _` → `true`、`false & _` → `false`、`true & x` → `x`、`false | x` → `x`（仅限 bool 常量）。条件可证明为常量（可靠界限）的 `WHERE` 选择对应分支；`WHERE(_, t, t)` → `t`；`WHERE(x, true, false)` → `x`；`WHERE(x, false, true)` → `!x`；`WHERE(a, WHERE(b, c, d), d)` → `WHERE(a & b, c, d)`。`dead_loop_patterns`：`vmax < 0` 的 `RANGE` → `CONST(0)`，`vmin == vmax` 的 `RANGE(CONST)` → 该常量。这里没有针对 `END`/`REDUCE` 空 range 的折叠；那由 `reduce_to_acc` 处理。

## 第 2 层

### 交换律规范化

对于**结果** dtype 为 `WeakInt` 的 `Add, Mul, Max, And, Or, Xor`（名义上还有 `Eq`/`Ne`，但它们的结果是 `Bool`，因而从不触发）：当 `tinygrad_tuplize_cmp(b, a) == Less` 时交换操作数，即线性化器也使用的结构化 `(op, arg, dtype, *src)` 键序。于是，在交换律意义下相等的索引表达式会哈希共享为同一个节点，重组规则和 expander 都依赖这一点。其他 dtype 保持编写时的顺序。

### 项合并（带常量的 `Add`/`Mul`）

| 模式 | 结果 |
|---------|--------|
| `Add(x, x)` | `x * 2` |
| `Add(Mul[x, c1], Mul[x, c2])` | `x * (c1 + c2)` |
| `Add[x, Mul[x, c]]` | `x * (c + 1)` |
| `Add[Add[y, Mul[x, c0]], Mul[x, c1]]` | `y + x * (c0 + c1)` |
| `Add[Add[y, x], Mul[x, c]]`、`Add[Add[y, Mul[x, c]], x]` | `y + x * (c + 1)` |
| `Add[Add[y, x], x]` | `y + x * 2` |
| `Mul[-1, Add[x, c]]` | `-x + (-c)` |
| `Mul[c, Add[x, k]]`，`x: WeakInt` | `c*x + c*k` |

### 布尔（`boolean_dsl_patterns`）

`Or[x, Not(x)]` → `true`，`And[x, Not(x)]` → `false`（仅限 bool）；两个方向的德摩根律，`And[Not(x), Not(y)]` → `!(x | y)` 以及 `Or[Not(x), Not(y)]` → `!(x & y)`。

### WHERE

`dce_dsl_patterns`：`WHERE(Not(c), t, f)` → `WHERE(c, f, t)`，除非 `f` 含有 `Invalid`（标量或某个 `STACK` lane）—— 交换会把标记移到真分支，gate 规则就看不到它了。`where_alu_combining_patterns`：当两个真分支或两个假分支都是常量时，对 `Add, Mul, Sub, Max, And, Or, Xor`，`op(WHERE(c, a, b), WHERE(c, d, e))` → `WHERE(c, op(a, d), op(b, e))`，还有结合形式 `Add(Add(y, WHERE(c, ..)), WHERE(c, ..))`。`where_bound_patterns`：`WHERE(Lt(x, c), t, f)` 在 `x.vmax < c.vmin` 时 → `t`，在 `x.vmin >= c.vmax` 时 → `f`。

### 界限坍缩与 min/max

`vmin_vmax_collapse_patterns`：可靠界限只有单一取值的 `Mul`、`FloorDiv`、`FloorMod`、比较、`PARAM` 或 `SPECIAL` 变成该常量（排除浮点；有意排除 `Add`/`Sub`/`Max`，以免只迭代一次的循环携带值被折叠掉）。`minmax_dsl_patterns`：当 `x.vmin >= y.vmax` 时 `Max(x, y)` → `x`（浮点要求严格大于，以保留零的符号），对 `y` 对称。没有 `Min` 算子：`Tensor::minimum` 是一个 `WHERE`。

### ALU 链折叠（`alu_folding_dsl_patterns`）

结合律折叠 `(x ⊕ c1) ⊕ c2` → `x ⊕ (c1 ⊕ c2)`，适用于 `Add`、`Mul`、`And`、`Or`、`Xor`、`Max`；当 `y` 不是常量时，常量外推 `(x + c) + y` → `(x + y) + c` 和 `(x * c) * y` → `(x * y) * c`；`(x - c1) + c2`、`(x + c1) - c2` 规范化为 `x + k` 或 `x - |k|`；`(x - c1) - c2` → `x - (c1 + c2)`；`Sub(a, Sub(b, x))` → `x + (a - b)`。`Sub` 在 Svod 中是一等算子（Tinygrad 把 `a - b` 写作 `a + b*-1`）。

### 比较（`comparison_dsl_patterns`）

对全部六种比较：非浮点上的 `x op x` 被折叠（`Lt/Gt/Ne` → `false`，`Le/Ge/Eq` → `true`）；常量操作数被折叠；否则由 `ComparisonAnalyzer::analyze`（`ir/src/uop/comparison_analysis.rs`）根据可靠界限证明 `true` 或 `false` —— 两者都仅限非弱 dtype。然后：`Lt(Add[c0, x], c1)` → `Lt(x, c1 - c0)`；`Lt(Mul[x, -1], Mul[y, -1])` → `Lt(y, x)`；在无回绕检查下，对 `d > 0` 有 `Lt(FloorDiv(x, d), c)` → `Lt(x, c * d)`（对向下取整除法精确成立，`c` 可为任意符号）；对 `WeakInt`：`Lt(Mul[c0, x], c1)` → `±x < ceil(c1 / |c0|)`，以及 GCD 折叠 `lt_folding`（`x = d*q + r`，`r ∈ [0, d)`，`d | c` ⇒ `x < c ⇔ q < c/d`）。

### Range 与除法

`range_based_mod_div_patterns` 和 `advanced_division_dsl_patterns` 属于索引代数，见[索引算术](./index-arithmetic.md)页面。`range_based_cast_patterns` 对界限能放入 `a` 的强整数 `x` 坍缩 `CAST(CAST(x, a), b)`。`long_to_int_narrowing_patterns` 把操作数和结果都能放入 `i32` 的 `Int64` 二元算子重写为 `Int32` 算子再转换回来，并把有符号整数类型转换分配到 `WeakInt + c` 上。

### AFTER

`after_simplification_patterns`：不是副作用的 deps（`RANGE`、`STORE`、`END`、`CALL`、`BARRIER`、`CUSTOM`、`FUNCTION`）被替换为它们自己的源并去重；`NOOP` deps 和 `END(NOOP)` 链被丢弃；`AFTER(x, [])` → `x`。

## 第 3 层（`sym`）

- **`pm_simplify_valid`**（`valid_simplification.rs`）：由 `Bool` 有效性子句构成的 `And` 链逐子句化简（`simplify_valid`），而 `x: WeakInt` 的 `WHERE(cond, x, Invalid)` 会在 `cond` 蕴含的界限下重写 `x`（`uop_given_valid`）：`parse_valid` 把每个子句读作 `expr < c` / `expr >= c`，代入一个有界变量后重新化简。
- **`alu_vectorize_reorder_patterns`**：对 13 个算术/位运算算子和六种比较，当两个操作数都是同一节点、lane 数相同且 > 1 的广播时，`op(STACK(x, x, ..), STACK(y, y, ..))` → `STACK(op(x, y), ..)`。
- **`ne_zero_fold_patterns`**：`Ne(x, 0)` → `x.cast(bool)`。
- **`cast_where_dsl_patterns`**：`CAST(WHERE(s, a, b))` → `WHERE(s, CAST(a), CAST(b))`。
- **`store_load_folding_patterns`**：`STORE(_, Invalid)` → `NOOP`；`STORE(INDEX, WHERE(c, v, Invalid))` → 向带 gate 的索引 `INDEX(buf, WHERE(c, idx, Invalid))` 存储 `v`；`STORE(idx, LOAD(idx))` → `NOOP`；`STORE(INDEX, WHERE(g, alt, LOAD(same INDEX)))` → 带 gate 地存储 `alt`。
- **`reduce_sym_patterns`**：`REDUCE(x * c, Add)` → `REDUCE(x, Add) * c`，以及 `reduce_mul_chain_sym`（把与 range 无关的因子提出 `Add`/`Max` 归约；对 `Max` 只提出非负因子），仅限整数。
- **`sym_phase3_patterns`**：`-1 * (x + y)` → `-x + -y`；对 `WeakInt`，`(x + y) * c` → `x*c + y*c`；单源 `GROUP` 解包；把 `NOOP`/`STACK`/`SINK` 展平进 `SINK`/`GROUP`；`END(NOOP)` → `NOOP`。

## 级联示例

`x: Int32` 时的 `(x + 0) * 1 + (3 + 4)`：

```text
Add(x, 0)        → x          identity_and_zero
Mul(x, 1)        → x          identity_and_zero
Add(3, 4)        → 7          constant_folding
Add(x, 7)                     stays: no rule
```

引擎先重写子节点再重写父节点，并对重建后的父节点重新匹配，因此这三步在一次 `graph_rewrite` 中完成。其中的恒等步骤正是 [Z3 预言测试](./pattern-system.md#verifying-rewrites-with-z3)所证明的那些。
