---
sidebar_label: Алгебраическое упрощение
---

# Алгебраическое упрощение

Символьный упроститель — это три вложенных матчера в `schedule/src/symbolic/patterns.rs`. Какой уровень где выполняется:

| Матчер | Состав | Где выполняется |
|---------|-------------|---------|
| `symbolic_simple()` | `symbolic_simple_base() + dead_loop_patterns()` | add-loads (13), devectorize (14), проход изображений (17), понижение индексов (17), ранние декомпозиции (19b) и набор `pm_decomp` внутри финальной перезаписи |
| `symbolic()` | `symbolic_simple` + группы уровня 2 | мега-проход rangeify, разбиение/слияние диапазонов (`+ pm_fold_cast_const`), `indexing_simplify`, финальное символьное упрощение (18) |
| `sym()` | `symbolic` + уровень 3 | предоптимизация (`+ pm_fold_cast_const + pm_flatten_range`), post-opt символьное упрощение (08), раннее символьное упрощение (15), дополнительное символьное упрощение (16, `+ indexing_simplify`) |

`pm_fold_cast_const` (`CAST(CONST) → CONST`) намеренно *не* входит ни в один уровень; места, которым он нужен, добавляют его явно — ровно так же, как Tinygrad компонует `symbolic + pm_fold_cast_const` только там, где это делает `UOp.simplify`.

Каждое правило, переписывающее целочисленную арифметику, проходит через `exact_integer_rewrite` — типизированное доказательство отсутствия переполнения (`typed_integer_rewrite_is_exact`), которое отклоняет перезапись, если исходное выражение или замена могут переполнить свой конкретный dtype. Группы, помеченные как *зависящие от значений*, дополнительно обёрнуты в `value_sensitive`, который отключает их, пока для поддерева не выполнено `weak_float_values_are_committed`. Границы берутся из `VminVmaxProperty` (доступно всегда) и `SoundVminVmaxProperty` (`None` для операций, чьим границам нельзя доверять: загрузки, `Pow`, `Fdiv`); оба кешируются на узел.

## Состав

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

Порядок несущий: канонизация перед объединением слагаемых, свёртка ALU перед правилами сравнения и диапазонов, потому что каждая группа открывает сопоставления для следующей. Правила дистрибуции обратной величины из Tinygrad (`sym` в `uop/symbolic.py`) намеренно отсутствуют: все шесть неточны с точки зрения IEEE.

**Обозначения.** `OP[a, b]` — коммутативная, `OP(a, b)` — упорядоченная; `@zero`/`@one`/`c` — константы; повторяющееся имя — тот же узел (`Arc::ptr_eq`). `//` — это `FloorDiv`, `%` — `FloorMod`; усекающие `CDiv`/`CMod` появляются только после поздних декомпозиций.

## Уровень 1

### Распространение Invalid (`propagate_invalid`)

`Invalid` — это `UOp::invalid_marker()`, константа `ConstValue::Invalid`; `is_invalid_marker` также распознаёт `VCONST` или `STACK`, целиком состоящие из `Invalid`, и обёртки перемещения вокруг них. Валидность хранится как `WHERE(cond, x, Invalid)`, и эти правила сохраняют эту форму, пока вокруг неё перемещается арифметика:

| Паттерн | Результат |
|---------|--------|
| `WHERE(Invalid, _, _)` | `Invalid` |
| `WHERE(WHERE(c, x, Inv), a, b)` | `WHERE(c, WHERE(x, a, b), Inv)` |
| `WHERE(c, Inv, x)` | `WHERE(!c, x, Inv)` (`WHERE(c, Inv, Inv)` → `Inv`) |
| `WHERE(c1, WHERE(c2, x, d), d)` | `WHERE(c1 & c2, x, d)` |
| `WHERE(a, WHERE(c, x, Inv), y)`, `y` не `Invalid` | `WHERE(!a \| c, WHERE(a, x, y), Inv)` — и зеркально для ложной ветви |
| `unary(Inv)`, `CAST(Inv)`, `BITCAST(Inv)` | `Inv` |
| `unary(WHERE(c, x, Inv))`, `CAST`, `BITCAST` | `WHERE(c, unary(x), Inv)` |
| `op(WHERE(c, x, Inv), y)`, `op(y, WHERE(c, x, Inv))` для **каждой** бинарной операции, включая сравнения | `WHERE(c, op(x, y), Inv)` |
| `op(Inv, y)`, `op(y, Inv)` для 13 бинарных операций, не являющихся сравнениями | `Inv` |

Почему первым: `MUL(0, WHERE(c, x, Inv))` должен стать `WHERE(c, 0, Inv)`, а не `0`.

### Мёртвые загрузки и записи (`fold_invalid_load_store`)

`LOAD(INDEX(buf, Invalid, ..))` (также за `CAST`) → `alt` загрузки, если он есть, иначе ноль с формой; `STORE(INDEX(buf, Invalid, ..), v)` без гейта → `NOOP`.

### Свёртка констант

Унарные: `Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc` (`Neg` здесь не операция — `neg()` строит `MUL(x, -1)`). Бинарные: `Add, Mul, Sub, FloorMod, Max, Pow, FloorDiv, Fdiv, And, Or, Xor, Shl, Shr`, плюс шесть сравнений в `Bool`. Тернарные: `Where`, `MulAcc`. Результаты фиксируются через формат хранения dtype (`Int32` переполняется с заворачиванием), кроме слабых dtype, которые сохраняют неусечённое значение. `vconst_folding_patterns` делает то же самое по линиям для `VCONST ⊕ VCONST` и смеси broadcast `CONST`/`VCONST` (11 бинарных операций + сравнения, 6 унарных), пропуская слабые линии.

### Булева арифметика

`Mul[x, y]` → `x & y`, `Add[x, y]` → `x | y`, `Max(x, y)` → `x | y`, когда оба — `Bool`.

### Тождество и ноль

| Паттерн | Результат | Guard |
|---------|--------|-------|
| `Add[x, 0]` | `x` | не float, или ноль — это `-0.0` (`x + 0.0` не тождество для `x = -0.0`) |
| `Sub(x, 0)` | `x` | не float, или ноль — это `+0.0` |
| `Mul[x, 1]`, `Or[x, 0]`, `Xor[x, 0]`, `FloorDiv(x, 1)`, `Fdiv(x, 1)` | `x` | |
| `FloorMod(x, 1)` | `0` | |
| `Floor/Ceil/Trunc/Round(x)` | `x` | целый `x` |
| `Mul[x, 0]` | `0` | не float (`NaN * 0`, `Inf * 0` дают `NaN`) |
| `And[_, 0]` | `0` | |

### Свёртка с собой и с нулём

`FloorDiv(x, x)` → `1`; `FloorDiv(x, -1)` → `MUL(x, -1)`; `FloorMod(FloorMod(x, y), y)` → `FloorMod(x, y)`; `And(x, x)`, `Or(x, x)`, `Max(x, x)` → `x`; `FloorMod(x, x)` → `0`; `Lt(x, x)` → `false` для не-float, а для float — когда надёжные границы доказывают, что `x` не `NaN`; `Ne(x, x)` → `false` для целых и bool.

### Деление

`Fdiv(0.0, 0.0)` и `Fdiv(MUL[_, 0.0], 0.0)` → `NaN` (стоят первыми, чтобы опередить следующее правило); `Fdiv(x, x)` → `1.0` только когда `x` доказуемо конечен и ненулевой; `FloorDiv(Mul(x, y), y)` → `x`. Для float правила `(x*y)/y → x` нет.

### Приведения типов

`CAST(x, dt)` → `x`, когда dtype уже совпадает; `CAST(CAST(x, a), b)` → `x`, когда `x: b` и `can_safe_cast(b, a)` (`a` вмещает любое значение `b`: та же знаковость и не меньшая ширина, беззнаковый→знаковый требует одного дополнительного бита, float↔int — никогда); `CAST(CAST(x, a), b)` → `CAST(x, b)`, когда `a` не сужает `x`. `uint_pack_dsl_patterns` отменяет упаковку `(hi.cast(u64) << 32) | lo.cast(u64)`, которую строит Threefry, чтобы ГПСЧ оставался в 32-битном ALU.

### Рекомбинация div-mod

Одно правило на каждом `Add`: `fold_add_divmod_recombine`, порт из Tinygrad. Оно сплющивает цепочку `Add`, находит слагаемое `(base % div) * mul` и партнёра `q * (div * mul)`, у которого `q` — частное от чего-то, сравнимого с `base` по модулю `div` (`quotient_base`: `q == b // div`, возможно со слитыми `(x//c + a)//div` и сдвинутыми константами), и заменяет пару на `b * mul`; при `q == (b // div) % d` оно сворачивается в более широкое `(b % (div*d)) * mul`. Это семейство `x%n + (x//n)*n → x` и его масштабированные, смещённые и трёхчленные варианты, находимые через цепочку, а не отдельными правилами.

### Степени, булевы выражения, DCE

`Pow(x, 0)` → `1`, `Pow(x, 1)` → `x`, `Pow(1, x)` → `1` (только скаляры; другие показатели не переписываются — формы с обратной величиной и sqrt меняют округление IEEE). `Not(Not(x))` → `x`, `Xor(x, x)` → `0`, `true | _` → `true`, `false & _` → `false`, `true & x` → `x`, `false | x` → `x` (только bool-константы). `WHERE` с доказуемо константным условием (надёжные границы) выбирает ветвь; `WHERE(_, t, t)` → `t`; `WHERE(x, true, false)` → `x`; `WHERE(x, false, true)` → `!x`; `WHERE(a, WHERE(b, c, d), d)` → `WHERE(a & b, c, d)`. `dead_loop_patterns`: `RANGE` с `vmax < 0` → `CONST(0)`, `RANGE(CONST)` с `vmin == vmax` → эта константа. Свёртки `END`/`REDUCE` с пустым диапазоном здесь нет; такие случаи обрабатывает `reduce_to_acc`.

## Уровень 2

### Коммутативная канонизация

Для `Add, Mul, Max, And, Or, Xor` (и номинально `Eq`/`Ne`, которые никогда не срабатывают, потому что их результат — `Bool`), чей dtype **результата** — `WeakInt`: поменять операнды местами, если `tinygrad_tuplize_cmp(b, a) == Less`, — это структурный порядок ключа `(op, arg, dtype, *src)`, который использует и линеаризатор. Индексные выражения, равные с точностью до коммутативности, после этого хеш-консируются в один узел, на что полагаются рекомбинация и expander. Остальные dtype сохраняют авторский порядок.

### Объединение слагаемых (`Add`/`Mul` с константами)

| Паттерн | Результат |
|---------|--------|
| `Add(x, x)` | `x * 2` |
| `Add(Mul[x, c1], Mul[x, c2])` | `x * (c1 + c2)` |
| `Add[x, Mul[x, c]]` | `x * (c + 1)` |
| `Add[Add[y, Mul[x, c0]], Mul[x, c1]]` | `y + x * (c0 + c1)` |
| `Add[Add[y, x], Mul[x, c]]`, `Add[Add[y, Mul[x, c]], x]` | `y + x * (c + 1)` |
| `Add[Add[y, x], x]` | `y + x * 2` |
| `Mul[-1, Add[x, c]]` | `-x + (-c)` |
| `Mul[c, Add[x, k]]`, `x: WeakInt` | `c*x + c*k` |

### Булевы правила (`boolean_dsl_patterns`)

`Or[x, Not(x)]` → `true`, `And[x, Not(x)]` → `false` (только bool); законы де Моргана в обе стороны, `And[Not(x), Not(y)]` → `!(x | y)` и `Or[Not(x), Not(y)]` → `!(x & y)`.

### WHERE

`dce_dsl_patterns`: `WHERE(Not(c), t, f)` → `WHERE(c, f, t)`, если только `f` не содержит `Invalid` (скаляр или линию `STACK`) — перестановка перенесла бы маркер в истинную ветвь, где его не видят правила гейтов. `where_alu_combining_patterns`: `op(WHERE(c, a, b), WHERE(c, d, e))` → `WHERE(c, op(a, d), op(b, e))` для `Add, Mul, Sub, Max, And, Or, Xor`, когда обе истинные ветви или обе ложные ветви — константы, а также ассоциативная форма `Add(Add(y, WHERE(c, ..)), WHERE(c, ..))`. `where_bound_patterns`: `WHERE(Lt(x, c), t, f)` → `t`, когда `x.vmax < c.vmin`, и `f`, когда `x.vmin >= c.vmax`.

### Схлопывание по границам и min/max

`vmin_vmax_collapse_patterns`: `Mul`, `FloorDiv`, `FloorMod`, сравнение, `PARAM` или `SPECIAL`, чьи надёжные границы сводятся к одному значению, становятся этой константой (float исключены; `Add`/`Sub`/`Max` намеренно исключены, чтобы перенос значения в цикле с одной итерацией не был свёрнут). `minmax_dsl_patterns`: `Max(x, y)` → `x`, когда `x.vmin >= y.vmax` (для float — строго больше, чтобы сохранить знак нуля), симметрично для `y`. Операции `Min` нет: `Tensor::minimum` — это `WHERE`.

### Свёртка цепочек ALU (`alu_folding_dsl_patterns`)

Ассоциативная свёртка `(x ⊕ c1) ⊕ c2` → `x ⊕ (c1 ⊕ c2)` для `Add`, `Mul`, `And`, `Or`, `Xor`, `Max`; проталкивание констант `(x + c) + y` → `(x + y) + c` и `(x * c) * y` → `(x * y) * c`, когда `y` не константа; `(x - c1) + c2`, `(x + c1) - c2` нормализуются в `x + k` или `x - |k|`; `(x - c1) - c2` → `x - (c1 + c2)`; `Sub(a, Sub(b, x))` → `x + (a - b)`. `Sub` — полноценная операция в Svod (Tinygrad записывает `a - b` как `a + b*-1`).

### Сравнения (`comparison_dsl_patterns`)

Для всех шести сравнений: `x op x` на не-float сворачивается (`Lt/Gt/Ne` → `false`, `Le/Ge/Eq` → `true`); константные операнды сворачиваются; иначе `ComparisonAnalyzer::analyze` (`ir/src/uop/comparison_analysis.rs`) доказывает `true` или `false` по надёжным границам — оба только для неслабых dtype. Далее: `Lt(Add[c0, x], c1)` → `Lt(x, c1 - c0)`; `Lt(Mul[x, -1], Mul[y, -1])` → `Lt(y, x)`; `Lt(FloorDiv(x, d), c)` → `Lt(x, c * d)` для `d > 0` при проверке отсутствия переполнения (точно для деления с округлением вниз, при любом знаке `c`); для `WeakInt`: `Lt(Mul[c0, x], c1)` → `±x < ceil(c1 / |c0|)` и GCD-свёртка `lt_folding` (`x = d*q + r`, `r ∈ [0, d)`, `d | c` ⇒ `x < c ⇔ q < c/d`).

### Диапазоны и деление

`range_based_mod_div_patterns` и `advanced_division_dsl_patterns` — это алгебра индексов; они описаны на странице [индексной арифметики](./index-arithmetic.md). `range_based_cast_patterns` схлопывает `CAST(CAST(x, a), b)` для сильного целого `x`, чьи границы укладываются в `a`. `long_to_int_narrowing_patterns` переписывает бинарную операцию `Int64`, чьи операнды и результат укладываются в `i32`, как операцию `Int32` с обратным приведением, и распределяет знаковое целочисленное приведение по `WeakInt + c`.

### AFTER

`after_simplification_patterns`: зависимости, не являющиеся побочными эффектами (`RANGE`, `STORE`, `END`, `CALL`, `BARRIER`, `CUSTOM`, `FUNCTION`), заменяются их собственными источниками и дедуплицируются; зависимости `NOOP` и цепочки `END(NOOP)` отбрасываются; `AFTER(x, [])` → `x`.

## Уровень 3 (`sym`)

- **`pm_simplify_valid`** (`valid_simplification.rs`): цепочка `And` из клауз валидности `Bool` упрощается клауза за клаузой (`simplify_valid`), а `WHERE(cond, x, Invalid)` с `x: WeakInt` переписывает `x` при границах, которые следуют из `cond` (`uop_given_valid`): `parse_valid` читает каждую клаузу как `expr < c` / `expr >= c`, подставляет ограниченную переменную и упрощает заново.
- **`alu_vectorize_reorder_patterns`**: `op(STACK(x, x, ..), STACK(y, y, ..))` → `STACK(op(x, y), ..)` для 13 арифметических/битовых операций и шести сравнений, когда оба операнда — broadcast одного узла с одинаковым числом линий > 1.
- **`ne_zero_fold_patterns`**: `Ne(x, 0)` → `x.cast(bool)`.
- **`cast_where_dsl_patterns`**: `CAST(WHERE(s, a, b))` → `WHERE(s, CAST(a), CAST(b))`.
- **`store_load_folding_patterns`**: `STORE(_, Invalid)` → `NOOP`; `STORE(INDEX, WHERE(c, v, Invalid))` → индекс с гейтом `INDEX(buf, WHERE(c, idx, Invalid))`, записывающий `v`; `STORE(idx, LOAD(idx))` → `NOOP`; `STORE(INDEX, WHERE(g, alt, LOAD(same INDEX)))` → запись `alt` с гейтом.
- **`reduce_sym_patterns`**: `REDUCE(x * c, Add)` → `REDUCE(x, Add) * c` и `reduce_mul_chain_sym` (множители, не зависящие от диапазона, выносятся из редукции `Add`/`Max`; для `Max` — только неотрицательные), только для целых.
- **`sym_phase3_patterns`**: `-1 * (x + y)` → `-x + -y`; `(x + y) * c` → `x*c + y*c` для `WeakInt`; разворачивание `GROUP` с одним источником; сплющивание `NOOP`/`STACK`/`SINK` в `SINK`/`GROUP`; `END(NOOP)` → `NOOP`.

## Пример каскада

`(x + 0) * 1 + (3 + 4)` с `x: Int32`:

```text
Add(x, 0)        → x          identity_and_zero
Mul(x, 1)        → x          identity_and_zero
Add(3, 4)        → 7          constant_folding
Add(x, 7)                     stays: no rule
```

Движок переписывает потомков раньше родителей и заново сопоставляет пересобранного родителя, поэтому все три шага происходят за один `graph_rewrite`. Шаги с тождествами — именно те, что доказывают [оракулы Z3](./pattern-system.md#verifying-rewrites-with-z3).
