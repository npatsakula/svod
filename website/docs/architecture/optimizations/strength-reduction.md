---
sidebar_label: Strength reduction
---

# Strength Reduction and Late Decompositions

The late rewrites replace operations with cheaper equivalents and operations the backend lacks with ones it has. They run after index lowering, in stages `19b`–`20` of the [post-optimization pipeline](../codegen/linearizer.md), because the earlier passes need the original structure: `Add(Mul(a, b), c)` must stay visible to term combining before it becomes `MulAcc`. Source: `early_decomposition_patterns`, `get_late_rewrite_patterns`, `pm_mod_to_idiv` in `schedule/src/optimizer/mod.rs`; the rules in `schedule/src/rangeify/patterns.rs` and `schedule/src/symbolic/fast_div.rs`; `ir/src/decompositions/`. Tinygrad: `codegen/decomp/op.py` (`get_late_rewrite_patterns`, `fast_idiv`).

## Composition

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

`supports(..)` is the renderer's `RendererOps` table. Everything is one fixpoint per stage, so the rules feed each other: `pm_mul_to_shl` turns `R1 * 64` into `R1 << 6`, and `pm_shl_add_to_mulacc` then fuses `(R0 << 2) + (R1 << 6)` into `MulAcc(R0, 4, R1 << 6)` — the integer FMA in the [worked example](../codegen/worked-example.md). `DISABLE_FAST_IDIV` defaults to **1**: the magic-number division below is opt-in.

## Floor to truncating division

`divmod_decomposition_patterns` (`ir/src/decompositions/mod.rs`) lowers `FloorDiv`/`FloorMod` to the C-style `CDiv`/`CMod` every backend has, adding the sign correction `q - (r != 0 && (a < 0) != (b < 0))` / `r + (correction ? b : 0)` unless both operands provably sit on one side of zero (`same_truncating_bucket`). All power-of-two and magic-number rules below match `CDiv`/`CMod` (or `FloorMod`, for `pm_mod_to_and`, which also runs in `19b` so power-of-two modulos fold before the lowering).

## Power-of-two rules

| Rule | Pattern | Result | Guard |
|------|---------|--------|-------|
| `pm_mod_to_and` | `FloorMod(x, 2^n)` | `x & (2^n - 1)` | integer `x` (exact for floor modulo, any sign) |
| `pm_mul_to_shl` | `Mul[x, 2^n]` | `x << n` | integer `x` |
| `pm_div_to_shr` | `CDiv(x, 2^n)` | `x >> n` | `vmin(x) >= 0` or unsigned |
| | | `(x + WHERE(x < 0, 2^n - 1, 0)) >> n` | signed `x` that may be negative |

The bias corrects the arithmetic shift's rounding toward −∞ to the truncating division's rounding toward zero. On the LLVM backend a signed `Shr` renders as `ashr`, so the bias is required whenever `vmin` cannot be proven.

## Magic-number division (`fast_division_patterns`, `fast_div.rs`)

For `CDiv(x, d)` with a positive non-power-of-two constant `d` and `x` unsigned or `vmin(x) >= 0`:

1. `magic_unsigned(vmax, d)` — Hacker's Delight: `nc = (vmax + 1) / d * d - 1`, `nbits = 64 - leading_zeros(vmax)`, and the smallest `s ∈ 0..=2*nbits` with `2^s > nc * (d - 1 - (2^s - 1) % d)`; `M = (2^s + d - 1 - (2^s - 1) % d) / d`. The result `(x * M) >> s` equals `x / d` for all `0 <= x <= vmax`. The call uses `max(vmax, |vmin|)`.
2. If `M * vmin` and `M * vmax` fit the dtype: emit `(x * M) >> s`.
3. Else take the power-of-two factor out: `d = 2^k * d'` becomes `CDiv(x, 2^k)` (a shift after the previous rule) and recurse on `d'` without widening.
4. Else widen to the next integer dtype (`i8 → i16 → i32 → i64 → u64`, `u8 → u16 → u32 → u64`) when the renderer supports it and the product fits there, and cast back.

`pm_mod_to_idiv` then rewrites the matching `CMod(x, d)` as `x - d * CDiv(x, d)` so the remainder goes through the same path. A signed correction (`+ (x < 0)`) exists in `fast_idiv` but the pattern guard makes it unreachable. Example: `x ∈ [0, 255]`, `d = 7` → `M = 293`, `s = 11`; `(255 * 293) >> 11 = 36 = 255 / 7`.

## Float and FMA

- `pm_fdiv_to_mul`: `Fdiv(x, c)` → `x * (1/c)` for a float constant with `c != 0` and a finite reciprocal.
- `pm_fma_decomposition`: `Add[Mul(a, b), c]` → `MulAcc(a, b, c)` when all three share one float dtype. Integers are not fused here.
- `pm_shl_add_to_mulacc`: `Add[Shl(x, n), c]` → `MulAcc(x, 2^n, c)` — no float guard, so this is the integer path (`0 <= n < 64`).
- `pm_neg_from_mul`: `Mul[x, -1]` → `Neg(x)` (the only place a `Neg` op is created; `neg()` elsewhere builds `MUL(x, -1)`), and `Add[x, Neg(y)]` → `Sub(x, y)`.
- `pm_half_bf16_cast`: a same-width float cast (`f16 ↔ bf16`) has no single LLVM instruction and a plain `cast(f32).cast(dst)` chain would be folded back by the cast rules, so it is spelled through bits: `f16 → f32 → RNE-round the low 16 bits → bf16`, and `bf16 → (u16 << 16 as f32) → f16`.

## Comparison negations (`pm_comparison_negations`)

Integers only; the constant arithmetic uses `checked_*` and declines on overflow.

| Pattern | Result |
|---------|--------|
| `Not(Lt(x, c))` | `Lt(c - 1, x)` |
| `Not(Lt(c, x))` | `Lt(x, c + 1)` |
| `And[Lt(c1, x), Lt(x, c2)]`, `c2 == c1 + 2` | `Eq(x, c1 + 1)` |
| `Lt(Mul(x, -1), c)` | `Lt(-c, x)` |
| `Lt(Mul(x, -1), Mul(y, c))` | `Lt(y * -c, x)` |

`pm_demorgan` is the late `And[Not(x), Not(y)]` → `Not(Or(x, y))`, bool only, gated on `Or`; both De Morgan directions also live in `symbolic()`'s `boolean_dsl_patterns`, which `symbolic_simple` (and therefore the late fixpoint's own tier-1 set) does not include.

## Op decompositions

- `pm_max_decomposition`: `Max(a, b)` → `WHERE(a < b, b, a)`.
- `pm_erf_decomposition`: Abramowitz–Stegun 7.1.26, `erf(x) = sign(x) * (1 - t * P(t) * exp(-x²))` with `t = 1 / (1 + 0.3275911 |x|)` and `P` the Horner polynomial `1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592`; maximum error about 1.5e-7. `Erf` stays a UOp until here because `@llvm.erf` is a libm call the in-process JIT does not link.
- `pm_threefry_decomp`: Threefry2x32, five rounds in `u32` arithmetic.
- `get_transcendental_patterns`: `Exp2`/`Log2`/`Sin` → `xexp2`/`xlog2`/`xsin` (`ir/src/decompositions/transcendentals.rs`) for f16/f32/f64, other floats routed through f32; `Sqrt` → `xpow(x, 0.5)`; each only when the renderer lacks the op, all of them when `TRANSCENDENTAL=2`.
- Device `decompositor()` (Metal: `amd_decomposition_patterns` — `Exp`, `Log`, `Cos`, `Tan`, binary `Pow` over native `exp2`/`log2`).

## Dtype emulation (`19c`)

Between the early and late decompositions, `pm_dtype_decomp_commit` emulates dtypes the renderer does not support: `Int64`/`UInt64` as pairs of 32-bit words (`pm_long_decomp`, with carries from `Lt` and a 64-step shift-subtract divider for `CDiv`/`CMod`), and FP8/`Float16`/`BFloat16` as `Float16` or `Float32` compute over the original storage word (`pm_float_decomp`, bit-exact `f2f` conversions). The selection is made per graph by walking it once (`DTypeDecompCtx`); `get_dtype_decomps` exposes the same list for the compile cache key.
