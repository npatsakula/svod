---
sidebar_label: 强度削减
---

# 强度削减与后期分解

后期重写把操作替换为更廉价的等价形式，并把后端缺少的操作替换为它具备的操作。它们在索引降级之后运行，即 [post-optimization 流水线](../codegen/linearizer.md)的 `19b`–`20` 阶段，因为更早的 pass 需要原始结构：`Add(Mul(a, b), c)` 在变成 `MulAcc` 之前必须对项合并保持可见。源码：`schedule/src/optimizer/mod.rs` 中的 `early_decomposition_patterns`、`get_late_rewrite_patterns`、`pm_mod_to_idiv`；`schedule/src/rangeify/patterns.rs` 和 `schedule/src/symbolic/fast_div.rs` 中的规则；`ir/src/decompositions/`。Tinygrad：`codegen/decomp/op.py`（`get_late_rewrite_patterns`、`fast_idiv`）。

## 组合

```text
19b  early_decomposition_patterns(supported)
       symbolic_simple + pm_fold_cast_const + pm_mod_to_and + divmod_decomposition_patterns
       + pm_threefry_decomp              if !supports(Threefry)
       + pm_max_decomposition            if !supports(Max) && supports(Lt)
       + pm_erf_decomposition            if !supports(Erf)

19d  pm_decomp = early
       + get_late_rewrite_patterns(renderer, disable_fast_idiv)
           pm_mod_to_and + pm_half_bf16_cast
           + pm_demorgan                   if supports(Or)
           + pm_mul_to_shl                 if supports(Shl)
           + pm_div_to_shr                 if supports(Shr)
             + fast_division_patterns + pm_mod_to_idiv    if also DISABLE_FAST_IDIV=0
           + pm_neg_from_mul               if supports(Neg)
           + pm_comparison_negations       if supports(Lt) || supports(Eq)
           + pm_fma_decomposition          if supports(MulAcc)
             + pm_shl_add_to_mulacc        if also supports(Shl)
           + pm_fdiv_to_mul                if supports(Fdiv)
       + get_transcendental_patterns(supported, TRANSCENDENTAL >= 2)
       + renderer.decomposition_matcher()  if the device defines one

20   pm_final = pm_commit_weak + pm_cast_weak + pm_decomp (+ extra_matcher) + pm_split_ends
```

`supports(..)` 是 renderer 的 `RendererOps` 表。每个阶段都是一次不动点迭代，因此各规则相互促进：`pm_mul_to_shl` 把 `R1 * 64` 变成 `R1 << 6`，然后 `pm_shl_add_to_mulacc` 把 `(R0 << 2) + (R1 << 6)` 融合为 `MulAcc(R0, 4, R1 << 6)` —— 即[完整示例](../codegen/worked-example.md)中的整数 FMA。`DISABLE_FAST_IDIV` 默认为 **1**：下面的魔数除法需要主动开启。

## 向下取整除法转截断除法

`divmod_decomposition_patterns`（`ir/src/decompositions/mod.rs`）把 `FloorDiv`/`FloorMod` 降级为每个后端都具备的 C 风格 `CDiv`/`CMod`，并添加符号修正 `q - (r != 0 && (a < 0) != (b < 0))` / `r + (correction ? b : 0)`，除非两个操作数可证明位于零的同一侧（`same_truncating_bucket`）。下面所有 2 的幂规则和魔数规则都匹配 `CDiv`/`CMod`（`pm_mod_to_and` 则匹配 `FloorMod`，它也在 `19b` 中运行，使 2 的幂取模在降级之前就被折叠）。

## 2 的幂规则

| 规则 | 模式 | 结果 | 条件 |
|------|---------|--------|-------|
| `pm_mod_to_and` | `FloorMod(x, 2^n)` | `x & (2^n - 1)` | 整数 `x`（对向下取整取模精确成立，任意符号） |
| `pm_mul_to_shl` | `Mul[x, 2^n]` | `x << n` | 整数 `x` |
| `pm_div_to_shr` | `CDiv(x, 2^n)` | `x >> n` | `vmin(x) >= 0` 或无符号 |
| | | `(x + WHERE(x < 0, 2^n - 1, 0)) >> n` | 可能为负的有符号 `x` |

偏置把算术移位向 −∞ 的舍入修正为截断除法向零的舍入。在 LLVM 后端上，有符号 `Shr` 渲染为 `ashr`，因此只要无法证明 `vmin`，就必须加偏置。

## 魔数除法（`fast_division_patterns`，`fast_div.rs`）

对于 `CDiv(x, d)`，其中 `d` 是正的非 2 的幂常量，且 `x` 无符号或 `vmin(x) >= 0`：

1. `magic_unsigned(vmax, d)` —— 出自 Hacker's Delight：`nc = (vmax + 1) / d * d - 1`，`nbits = 64 - leading_zeros(vmax)`，以及满足 `2^s > nc * (d - 1 - (2^s - 1) % d)` 的最小 `s ∈ 0..=2*nbits`；`M = (2^s + d - 1 - (2^s - 1) % d) / d`。对所有 `0 <= x <= vmax`，结果 `(x * M) >> s` 都等于 `x / d`。调用时使用 `max(vmax, |vmin|)`。
2. 若 `M * vmin` 和 `M * vmax` 能放入该 dtype：生成 `(x * M) >> s`。
3. 否则提出 2 的幂因子：`d = 2^k * d'` 变成 `CDiv(x, 2^k)`（经前一条规则后成为移位），并在不加宽的情况下对 `d'` 递归。
4. 否则，若 renderer 支持且乘积能放入下一个整数 dtype，就加宽到该 dtype（`i8 → i16 → i32 → i64 → u64`、`u8 → u16 → u32 → u64`），再转换回来。

`pm_mod_to_idiv` 随后把对应的 `CMod(x, d)` 重写为 `x - d * CDiv(x, d)`，使余数走同一条路径。`fast_idiv` 中存在有符号修正（`+ (x < 0)`），但模式的 guard 使其不可达。示例：`x ∈ [0, 255]`、`d = 7` → `M = 293`、`s = 11`；`(255 * 293) >> 11 = 36 = 255 / 7`。

## 浮点与 FMA

- `pm_fdiv_to_mul`：对于满足 `c != 0` 且倒数有限的浮点常量，`Fdiv(x, c)` → `x * (1/c)`。
- `pm_fma_decomposition`：当三者共享同一浮点 dtype 时，`Add[Mul(a, b), c]` → `MulAcc(a, b, c)`。整数不在这里融合。
- `pm_shl_add_to_mulacc`：`Add[Shl(x, n), c]` → `MulAcc(x, 2^n, c)` —— 没有浮点条件，所以这是整数路径（`0 <= n < 64`）。
- `pm_neg_from_mul`：`Mul[x, -1]` → `Neg(x)`（这是唯一创建 `Neg` 算子的地方；其他地方的 `neg()` 构建的是 `MUL(x, -1)`），以及 `Add[x, Neg(y)]` → `Sub(x, y)`。
- `pm_half_bf16_cast`：同宽度浮点转换（`f16 ↔ bf16`）没有单条 LLVM 指令，而普通的 `cast(f32).cast(dst)` 链又会被类型转换规则折叠回去，因此通过位操作来表达：`f16 → f32 → RNE-round the low 16 bits → bf16`，以及 `bf16 → (u16 << 16 as f32) → f16`。

## 比较取反（`pm_comparison_negations`）

仅限整数；常量运算使用 `checked_*`，溢出时放弃。

| 模式 | 结果 |
|---------|--------|
| `Not(Lt(x, c))` | `Lt(c - 1, x)` |
| `Not(Lt(c, x))` | `Lt(x, c + 1)` |
| `And[Lt(c1, x), Lt(x, c2)]`，`c2 == c1 + 2` | `Eq(x, c1 + 1)` |
| `Lt(Mul(x, -1), c)` | `Lt(-c, x)` |
| `Lt(Mul(x, -1), Mul(y, c))` | `Lt(y * -c, x)` |

`pm_demorgan` 是后期的 `And[Not(x), Not(y)]` → `Not(Or(x, y))`，仅限 bool，受 `Or` 能力控制；两个方向的德摩根律也都存在于 `symbolic()` 的 `boolean_dsl_patterns` 中，而 `symbolic_simple`（因此也包括后期不动点自身的第 1 层规则集）并不包含它。

## 算子分解

- `pm_max_decomposition`：`Max(a, b)` → `WHERE(a < b, b, a)`。
- `pm_erf_decomposition`：Abramowitz–Stegun 7.1.26，`erf(x) = sign(x) * (1 - t * P(t) * exp(-x²))`，其中 `t = 1 / (1 + 0.3275911 |x|)`，`P` 是系数为 `1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592` 的 Horner 多项式；最大误差约 1.5e-7。`Erf` 一直保持为 UOp 直到这里，因为 `@llvm.erf` 是 libm 调用，而进程内 JIT 不会链接它。
- `pm_threefry_decomp`：Threefry2x32，以 `u32` 运算执行五轮。
- `get_transcendental_patterns`：针对 f16/f32/f64，`Exp2`/`Log2`/`Sin` → `xexp2`/`xlog2`/`xsin`（`ir/src/decompositions/transcendentals.rs`），其他浮点类型经由 f32；`Sqrt` → `xpow(x, 0.5)`；每项仅在 renderer 缺少该算子时启用，`TRANSCENDENTAL=2` 时全部启用。
- 设备的 `decompositor()`（Metal：`amd_decomposition_patterns` —— 基于原生 `exp2`/`log2` 实现 `Exp`、`Log`、`Cos`、`Tan`、二元 `Pow`）。

## dtype 模拟（`19c`）

在早期分解与后期分解之间，`pm_dtype_decomp_commit` 模拟 renderer 不支持的 dtype：`Int64`/`UInt64` 用 32 位字对表示（`pm_long_decomp`，进位来自 `Lt`，`CDiv`/`CMod` 使用 64 步移位-减法除法器），FP8/`Float16`/`BFloat16` 则在原始存储字之上用 `Float16` 或 `Float32` 计算（`pm_float_decomp`，逐位精确的 `f2f` 转换）。选择按图进行，只需遍历一次（`DTypeDecompCtx`）；`get_dtype_decomps` 为编译缓存键暴露同一列表。
