---
sidebar_label: Снижение стоимости операций
---

# Снижение стоимости операций и поздние декомпозиции

Поздние перезаписи заменяют операции более дешёвыми эквивалентами, а операции, которых нет у бэкенда, — теми, что у него есть. Они выполняются после понижения индексов, на стадиях `19b`–`20` [post-оптимизационного пайплайна](../codegen/linearizer.md), потому что более ранним проходам нужна исходная структура: `Add(Mul(a, b), c)` должен оставаться видимым для объединения слагаемых, прежде чем станет `MulAcc`. Исходники: `early_decomposition_patterns`, `get_late_rewrite_patterns`, `pm_mod_to_idiv` в `schedule/src/optimizer/mod.rs`; правила в `schedule/src/rangeify/patterns.rs` и `schedule/src/symbolic/fast_div.rs`; `ir/src/decompositions/`. Tinygrad: `codegen/decomp/op.py` (`get_late_rewrite_patterns`, `fast_idiv`).

## Состав

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

`supports(..)` — это таблица `RendererOps` рендерера. На каждой стадии всё выполняется в одной неподвижной точке, поэтому правила питают друг друга: `pm_mul_to_shl` превращает `R1 * 64` в `R1 << 6`, а затем `pm_shl_add_to_mulacc` сливает `(R0 << 2) + (R1 << 6)` в `MulAcc(R0, 4, R1 << 6)` — целочисленный FMA из [разобранного примера](../codegen/worked-example.md). `DISABLE_FAST_IDIV` по умолчанию равен **1**: деление через магическое число ниже включается явно.

## От деления с округлением вниз к усекающему

`divmod_decomposition_patterns` (`ir/src/decompositions/mod.rs`) понижает `FloorDiv`/`FloorMod` до `CDiv`/`CMod` в стиле C, которые есть у любого бэкенда, добавляя коррекцию знака `q - (r != 0 && (a < 0) != (b < 0))` / `r + (correction ? b : 0)`, если только оба операнда не лежат доказуемо по одну сторону от нуля (`same_truncating_bucket`). Все правила для степеней двойки и магических чисел ниже сопоставляют `CDiv`/`CMod` (или `FloorMod` — для `pm_mod_to_and`, который также выполняется на `19b`, чтобы остатки по степени двойки сворачивались до понижения).

## Правила для степеней двойки

| Правило | Паттерн | Результат | Guard |
|------|---------|--------|-------|
| `pm_mod_to_and` | `FloorMod(x, 2^n)` | `x & (2^n - 1)` | целый `x` (точно для остатка с округлением вниз, при любом знаке) |
| `pm_mul_to_shl` | `Mul[x, 2^n]` | `x << n` | целый `x` |
| `pm_div_to_shr` | `CDiv(x, 2^n)` | `x >> n` | `vmin(x) >= 0` или беззнаковый |
| | | `(x + WHERE(x < 0, 2^n - 1, 0)) >> n` | знаковый `x`, который может быть отрицательным |

Смещение исправляет округление арифметического сдвига к −∞ на округление усекающего деления к нулю. На бэкенде LLVM знаковый `Shr` рендерится как `ashr`, поэтому смещение требуется всегда, когда `vmin` нельзя доказать.

## Деление через магическое число (`fast_division_patterns`, `fast_div.rs`)

Для `CDiv(x, d)` с положительной константой `d`, не являющейся степенью двойки, и беззнаковым `x` или `vmin(x) >= 0`:

1. `magic_unsigned(vmax, d)` — Hacker's Delight: `nc = (vmax + 1) / d * d - 1`, `nbits = 64 - leading_zeros(vmax)` и наименьшее `s ∈ 0..=2*nbits` с `2^s > nc * (d - 1 - (2^s - 1) % d)`; `M = (2^s + d - 1 - (2^s - 1) % d) / d`. Результат `(x * M) >> s` равен `x / d` для всех `0 <= x <= vmax`. Вызов использует `max(vmax, |vmin|)`.
2. Если `M * vmin` и `M * vmax` укладываются в dtype: выдать `(x * M) >> s`.
3. Иначе вынести множитель-степень двойки: `d = 2^k * d'` становится `CDiv(x, 2^k)` (сдвиг после предыдущего правила) с рекурсией по `d'` без расширения.
4. Иначе расширить до следующего целочисленного dtype (`i8 → i16 → i32 → i64 → u64`, `u8 → u16 → u32 → u64`), если рендерер его поддерживает и произведение там укладывается, и привести обратно.

Затем `pm_mod_to_idiv` переписывает соответствующий `CMod(x, d)` как `x - d * CDiv(x, d)`, чтобы остаток шёл по тому же пути. Знаковая коррекция (`+ (x < 0)`) в `fast_idiv` существует, но guard паттерна делает её недостижимой. Пример: `x ∈ [0, 255]`, `d = 7` → `M = 293`, `s = 11`; `(255 * 293) >> 11 = 36 = 255 / 7`.

## Float и FMA

- `pm_fdiv_to_mul`: `Fdiv(x, c)` → `x * (1/c)` для float-константы с `c != 0` и конечной обратной величиной.
- `pm_fma_decomposition`: `Add[Mul(a, b), c]` → `MulAcc(a, b, c)`, когда все три имеют один float dtype. Целые здесь не сливаются.
- `pm_shl_add_to_mulacc`: `Add[Shl(x, n), c]` → `MulAcc(x, 2^n, c)` — без float-guard, так что это целочисленный путь (`0 <= n < 64`).
- `pm_neg_from_mul`: `Mul[x, -1]` → `Neg(x)` (единственное место, где создаётся операция `Neg`; `neg()` в остальных местах строит `MUL(x, -1)`), и `Add[x, Neg(y)]` → `Sub(x, y)`.
- `pm_half_bf16_cast`: для приведения float одинаковой ширины (`f16 ↔ bf16`) нет одной инструкции LLVM, а простая цепочка `cast(f32).cast(dst)` была бы свёрнута обратно правилами приведений, поэтому оно записывается через биты: `f16 → f32 → RNE-round the low 16 bits → bf16` и `bf16 → (u16 << 16 as f32) → f16`.

## Отрицания сравнений (`pm_comparison_negations`)

Только для целых; константная арифметика использует `checked_*` и отказывается при переполнении.

| Паттерн | Результат |
|---------|--------|
| `Not(Lt(x, c))` | `Lt(c - 1, x)` |
| `Not(Lt(c, x))` | `Lt(x, c + 1)` |
| `And[Lt(c1, x), Lt(x, c2)]`, `c2 == c1 + 2` | `Eq(x, c1 + 1)` |
| `Lt(Mul(x, -1), c)` | `Lt(-c, x)` |
| `Lt(Mul(x, -1), Mul(y, c))` | `Lt(y * -c, x)` |

`pm_demorgan` — это поздний `And[Not(x), Not(y)]` → `Not(Or(x, y))`, только для bool, при наличии `Or`; оба направления законов де Моргана есть и в `boolean_dsl_patterns` из `symbolic()`, который не входит в `symbolic_simple` (а значит, и в собственный набор уровня 1 поздней неподвижной точки).

## Декомпозиции операций

- `pm_max_decomposition`: `Max(a, b)` → `WHERE(a < b, b, a)`.
- `pm_erf_decomposition`: Абрамовиц–Стиган 7.1.26, `erf(x) = sign(x) * (1 - t * P(t) * exp(-x²))` с `t = 1 / (1 + 0.3275911 |x|)`, где `P` — полином Горнера `1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592`; максимальная погрешность около 1.5e-7. `Erf` остаётся UOp до этого места, потому что `@llvm.erf` — это вызов libm, который JIT в процессе не линкует.
- `pm_threefry_decomp`: Threefry2x32, пять раундов в арифметике `u32`.
- `get_transcendental_patterns`: `Exp2`/`Log2`/`Sin` → `xexp2`/`xlog2`/`xsin` (`ir/src/decompositions/transcendentals.rs`) для f16/f32/f64, остальные float направляются через f32; `Sqrt` → `xpow(x, 0.5)`; каждая — только когда у рендерера нет этой операции, все сразу — при `TRANSCENDENTAL=2`.
- `decompositor()` устройства (Metal: `amd_decomposition_patterns` — `Exp`, `Log`, `Cos`, `Tan`, бинарный `Pow` поверх нативных `exp2`/`log2`).

## Эмуляция dtype (`19c`)

Между ранними и поздними декомпозициями `pm_dtype_decomp_commit` эмулирует dtype, которые рендерер не поддерживает: `Int64`/`UInt64` как пары 32-битных слов (`pm_long_decomp`, с переносами из `Lt` и 64-шаговым делителем сдвигом и вычитанием для `CDiv`/`CMod`), а FP8/`Float16`/`BFloat16` — как вычисления в `Float16` или `Float32` над исходным словом хранения (`pm_float_decomp`, побитово точные преобразования `f2f`). Выбор делается для каждого графа за один его обход (`DTypeDecompCtx`); `get_dtype_decomps` предоставляет тот же список для ключа кеша компиляции.
