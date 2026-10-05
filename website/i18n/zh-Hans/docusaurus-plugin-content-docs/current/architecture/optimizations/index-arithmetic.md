---
sidebar_label: 索引算术
---

# 索引算术

形状为 `[H, W]` 的 `tensor[i, j]` 访问就是 `i * W + j`。经过 range 拆分、合并、展开和 GPU 维度重建之后，索引表达式会累积出 `FloorDiv`/`FloorMod` 链；一个残留到后端的除法要花费数十个周期，而地址计算的其余部分只需一个周期。本页介绍消除它们的整数代数。源码：`schedule/src/symbolic/patterns.rs` 中的 `range_based_mod_div_patterns`、`advanced_division_dsl_patterns` 和 `div_mod_recombine_dsl_patterns`，`schedule/src/symbolic/divmod.rs` 中的 `fold_divmod_general`，降级位于 `schedule/src/symbolic/index_lowering.rs`。Tinygrad：`uop/divandmod.py`、`uop/symbolic.py`、`uop/weak.py`。

每条规则都基于两个事实：

- 在阶段 17 确定类型之前，索引算术都是 `WeakInt`（`UOp::index_const` 构建的是 `WeakInt`）；`ScalarDType::Index` 是遗留名称，`spec` 会在降级后验证它不存在。
- 每次重写都经过 `exact_integer_rewrite`：只有当原表达式和替换表达式都不会在其具体 dtype 中回绕时，候选才被接受。辅助函数在无界整数代数中构造候选；证明在调用处完成。

## 基于 range 的规则（`range_based_mod_div_patterns`）

`SoundVminVmaxProperty` 给出每个节点的 `[vmin, vmax]`（`RANGE(end)` 为 `[0, end-1]`）：

| 模式 | 结果 | 条件 |
|---------|--------|-------|
| `RANGE(end) % end`、`RANGE(end) // end` | 该 range、`0` | 同一个 `end` 节点 |
| `x % n` | `x` | `0 <= vmin(x)` 且 `vmax(x) < n` |
| `(a*n + b) % n` | `b % n` | `vmin(b) >= 0`；`n` 按值比较 |
| `(a*n + b + c) % n` | `(b + c) % n` | `vmin(b + c) >= 0` |
| `(a*n + b) // n` | 当 `0 <= b < n` 时为 `a`，否则为 `a + b // n` | `n > 0`、`vmin(b) >= 0` |
| `x // n` | `k` | `vmin(x) // n == vmax(x) // n`（同一个桶） |
| `(a + (x//n)*n) // n` | `x // n` | `0 <= a < n` |
| `(x + c) // d` | `x // d` | `c > 0`、`d > 0`、`vmin(x) >= 0`，且 `x % d` 可能取到的最大余数加上 `c` 仍小于 `d` |
| `(x + c) // d` | `(x + c%d) // d + c//d` | `c % d != c`，任意 `d != 0` |
| `(x + c) // d`，`x <= 0 <= x + c` | `-(-(c%d + x - (d-1)) // d) + c//d` | `d > 0` |

前两行是主力：拆分之后，`RANGE(n) % n` 和 `(outer*4 + inner) // 4` 就是全部内容。`(x + c) // d` 规则中的余数界限是基于 `x` 的取值范围 `[vmin, vmax]` 计算的，而不是基于其步长，因此 `(R*4 + 1) // 8` 只有在 `R*4` 覆盖少于 8 个值时才会折叠。

## 高级除法（`advanced_division_dsl_patterns`）

| 模式 | 结果 | 条件 |
|---------|--------|-------|
| `(a // b) // c` | `a // (b*c)` | `b != 0`、`c > 0`，`b*c` 不回绕 |
| `expr // d` | `expr.divides(d)` | `expr` 的每个加法项都能被精确整除（`UOp::divides` 会递归穿过 `Add`，因此两项都可整除的 `(a + b) // c` 也被覆盖） |
| `(a + b) % c` | `b % c` 或 `a % c` | 被丢弃的项能被精确整除 |
| `(x + c) % d` | `(x + c%d) % d` | `d > 0`、`c % d != c` |
| `x % y`、`x // y` | `fold_divmod_general(..)` | 见下文 |
| `(a - b) // c` | `a//c - b//c` | 两者都能被精确整除 |
| `(a//c1 + c2) // c3` | `(a + c1*c2) // (c1*c3)` | `c1, c3 > 0`，`a` 与 `c2` 同号，乘积不回绕 |

整数上没有通用的 `c * (a + b)` 分配律：只有项合并和 `sym_phase3_patterns` 中的 `WeakInt` 形式和 `-1` 形式。

## `fold_divmod_general`

移植自 Tinygrad 的 `fold_divmod_general`，对任何分母取值范围不恰好为 `{0}` 的标量整数 `FloorDiv`/`FloorMod` 调用。规则按顺序尝试；第一个产生候选的规则胜出：

1. **cancel** —— 若商 `x // y` 的取值范围只有单一值，则返回它（对 `%`：`x - q*y`）。
2. **multiple-of guard** —— 一个被声明为 `multiple_of` 某个因子（且该除数整除这个因子）的 `PARAM`：`% → 0`，`//` 不变。
3. **nested_div**（仅 `//`）—— `(a % (k*c)) // c` → `(a // c) % k`，`k.vmin > 0`。
4. **remove_nested_mod**（仅 `%`）—— `(a % (k*c) + b) % c` → `(a + b) % c`。
5. **congruence**（`fold_divmod_congruence`）—— 写成 `x = Σ f_i t_i + k`；对每个系数选择一个余数 `r_i ≡ f_i (mod c)`（取两种符号中较小的那个；只有单项或恰好相等时两种都尝试，按 `itertools.product` 的顺序）；若 `rem = Σ r_i t_i + k%c` 落在同一个商桶内，则 `x % c = rem - bucket*c` 且 `x // c = Σ (f_i - r_i)/c · t_i + (k - k%c + bucket*c)/c`。与符号无关。
6. **gcd_with_remainder** —— 当 `g = gcd(c, all f_i) > 1` 且 `x/g` 非负时：`((x/g + shift) // (c/g))` 或 `((x/g + shift) % (c/g)) * g + k%g`。
7. **nest_by_factor** —— 对每个真整除 `c` 的系数 `f`（`2 <= f < c`），把 `x // c` 重写为 `(x // f) // (c/f)`（递归折叠内层除法），把 `%` 重写为 `((x // f) % (c/f)) * f + x % f`；`node_count()` 最小的候选胜出。`%` 分支需要 `x >= 0`，且低位数字可证明落在 `[0, f)` 中。
8. **divide_by_gcd**（任意分母）—— 当 `symbolic_gcd(terms, y)` 不为 1 时，`x op y` → `(x/g) op (y/g)`（对 `%` 再乘以 `g`）。符号 `N` 下的 `(N*i + j) // N` 就是靠它折叠的。
9. **factor_remainder**（任意分母）—— 把各项分为能被 `y` 精确整除的项和其余项；`//` → `quotient + rest // y`，`%` → `rest % y`；常量除数还会把系数约简为其余数（`(r*8 + v) % 7` → `(r + v) % 7`）。需要 `x >= 0`、`y >= 0` 且余数非负。

示例：`(R*8 + v) // 8`，其中 `R: [0, 16)`、`v: [0, 8)` —— 规则 1 失败（16 个桶），规则 5 为系数 `8` 给出余数 `0`、为 `v` 给出 `1`，`rem = v ∈ [0, 7]` 落在一个桶内，因此商为 `R`，余数为 `v`。

## 重组

`fold_add_divmod_recombine`，作用于每个 `Add`（第 1 层）：寻找一个缩放后的余数 `(base % div) * mul` 和伙伴项 `q * (div*mul)`，其中 `q` 是某个模 `div` 与 `base` 同余之量的商，并返回 `b * mul` 加上其余各项 —— 当 `q` 本身是 `(b // div) % d` 时则返回 `(b % (div*d)) * mul`。变体 `x%n + (x//n)*n → x`、`(x//a)%c + (x//(a*c))*c → x//a`、`(x%c1)*c2 + (x//c1)*(c1*c2) → x*c2` 以及各种偏移形式，都出自这一次搜索。

## 索引 dtype 降级

`pm_lower_index_dtype`（阶段 `17-pm_lower_index_dtype`，见 [Devectorizer 页面](../codegen/devectorizer.md)）把 `WeakInt` 确定为 `Int32` 或 `Int64`：

- `select_dtype(u)`：`WeakFloat` → 默认浮点类型；整数界限位于 `[i32::MIN, i32::MAX]` 之内 → 默认整数类型，否则为 `Int64`。
- 叶子（`CONST`、`VCONST`、标量 `PARAM` 变量）变成 `concrete.cast(WeakInt)`；`Unary`、`Binary`、`WHERE`、`RANGE`、`STACK`、`SPECIAL` 解开其源上的类型转换，并以具体 dtype 重建（二元算子取 `select_dtype(u)` 与各源的 `least_upper_dtype`）；弱类型 `INDEX` 转换其缓冲区并确定其索引；非弱类型的消费者在自己的边上吸收末尾的类型转换（`lower_weak_srcs`，按内核记忆化）。
- `WHERE(valid, idx, Invalid)` 保持其形状；`Invalid` 从不被转换。若带 gate 的索引最终为 `Int64`，当缓冲区元素数落在 `i32` 范围内时会被收窄回 `Int32`。

有效性 `WHERE` 在两个阶段之后、于 `pm_move_gates_from_index`（`late/gater.rs`）中变成 LOAD/STORE 的 `gate`；`INDEX` 本身没有 gate 字段。

## 完整示例

形状为 `[4, 8]` 的 `tensor[i, j]`，用 `R0 ∈ [0, 32)` 扁平地遍历 32 个元素：

```text
row = R0 // 8, col = R0 % 8
addr = row * 8 + col = (R0 // 8) * 8 + (R0 % 8)
```

重组立即给出 `addr = R0`。现在按 4 展开：`pm_split_ranges` 代入 `R0 = R1 * 4 + R2`，其中 `R1 ∈ [0, 8)`、`R2 ∈ [0, 4)`：

```text
row = (R1*4 + R2) // 8
col = (R1*4 + R2) % 8
```

对它们运行 `graph_rewrite(symbolic(), ..)`（下面的树就是其输出）：

```text
row = (R1*4 + R2)//8
[31] FloorDiv : Scalar(WeakInt) shape=[]
├── [13] RANGE(R1, Weak) : Scalar(WeakInt) shape=[]
│   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
└── [29] CONST(Int(2)) : Scalar(WeakInt) shape=[]

col = (R1*4 + R2)%8
[55] Add : Scalar(WeakInt) shape=[]
├── [54] Mul : Scalar(WeakInt) shape=[]
│   ├── [50] FloorMod : Scalar(WeakInt) shape=[]
│   │   ├── [13] RANGE(R1, Weak) : Scalar(WeakInt) shape=[]
│   │   │   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
│   │   └── [46] CONST(Int(2)) : Scalar(WeakInt) shape=[]
│   └── [14] CONST(Int(4)) : Scalar(WeakInt) shape=[]
└── [15] RANGE(R2, Weak) : Scalar(WeakInt) shape=[]
    └── [14] → (see above)
```

`row`：规则 1 失败（四个桶），congruence 失败（`rem = R1*4 + R2 ∈ [0, 31]` 跨越四个桶），`gcd(8, 4, 1) = 1`，而 `f = 4` 的 `nest_by_factor` 通过同余（`rem = R2 ∈ [0, 3]`）把 `(R1*4 + R2) // 4` 折叠为 `R1`，剩下 `R1 // 2`。`col`：同一个因子给出 `(R1 % 2) * 4 + R2`。合在一起，`row*8 + col` 重组为 `R1*4 + R2` —— 尽管 `row` 和 `col` 各自仍带有一次除法和一次取模，地址却再次变成线性的。再拆分一次，`R1 = R3 * 2 + R4`（`R3 ∈ [0, 4)`、`R4 ∈ [0, 2)`），基于 range 的规则即可完成剩下的工作：

```text
row = (R3*8 + R4*4 + R2) // 8   → R3                (a*n + b)//n with b = R4*4 + R2 ∈ [0, 7]
col = (R3*8 + R4*4 + R2) % 8    → R4*4 + R2         (a*n + b + c)%n, then x % n → x
addr = row*8 + col              → (R4*4 + R2) + R3*8
```

零次除法，零次取模：分块后的地址就是扁平地址，并由重写证明。同一次运行的另外两个输出：`(R*8 + v) // 8`，其中 `R: [0, 16)`、`v: [0, 8)` → `R`（规则 5），以及 `(R*8 + v) % 7` → `(R + v) % 7`（规则 9 的系数约简）。
