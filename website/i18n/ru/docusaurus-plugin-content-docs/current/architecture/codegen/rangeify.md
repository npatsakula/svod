---
sidebar_label: Rangeify и разрез на ядра
---

# Rangeify, разрез на ядра и предоптимизация

Всё, что описано на этой странице, выполняется до того, как оптимизатор увидит ядро. Исходники: `schedule/src/rangeify/` и `apply_pre_optimization` в `schedule/src/optimizer/mod.rs`.

## Rangeify (`rangeify_with_map`)

Вход: тензорный граф, построенный вызовом `realize()` (операции перемещения, `REDUCE` в тензорной форме с `num_axes > 0`, `CONTIGUOUS`, `COPY`, ...). Выход: граф, в котором каждый цикл — явный `RANGE`, каждая материализация — `STAGE`, а каждое чтение — `INDEX`.

Проходы по порядку (`rangeify/transforms.rs`):

1. **Разрешение мультиустройств** — `multi_pm`, затем `lower_allreduce_pm` (оба через `graph_rewrite_preserve_calls`); `validate_supported_subset` отклоняет то, что бэкенды не умеют исполнять.
2. **`add_tags_patterns`** (снизу вверх) нумерует каждый узел, который можно пометить, тегом `[i]`. Теги — это идентичность тензора: после разреза карта выходов восстанавливается по выжившим тегам. `PARAM`, `CONST`, `RANGE`, `END`, `CALL`, операции перемещения и `MSTACK`/`MSELECT`, состоящие только из `PARAM`, не помечаются.
3. **`resolve_calls`** подставляет тела `FUNCTION` вместо их аргументов и сворачивает `GETTUPLE(TUPLE(..), i)`. Предкомпилированные функции и аргументы `CALL` остаются непрозрачными.
4. **Самые ранние перезаписи** (снизу вверх, один матчер): `movement_op_patterns + early_rewrites + split_reduceop_patterns`. `early_rewrites` отбрасывает `DETACH`/`CONTIGUOUS_BACKWARD`, сливает непомеченные цепочки `RESHAPE`, расширяет целочисленные произведения под расширяющим приведением типа, материализует источник `COPY` с изменённым размером или порядком через `CONTIGUOUS`, удаляет `COPY` в пределах одного устройства и сворачивает тензоры нулевого размера в константы. `split_reduceop` — это двухстадийное разбиение редукции (см. [оптимизацию диапазонов](../optimizations/range-optimization.md)).
5. **`run_rangeify`** (`rangeify/indexing.rs`):
   - `pm_generate_realize_map` (снизу вверх): отмечает то, что должно стать буфером, — `STORE`, `CONTIGUOUS`, `COPY` и их несмежные источники, источники `MSTACK`/`MSELECT` и входы написанного вручную ядра `CALL` (закреплены как неудаляемые).
   - `assign_ranges`: обход от корня к листьям. Материализуемый узел получает свежие диапазоны `Weak` для каждого выходного измерения (`IndexingContext::new_range`; измерение размера 1 — это `CONST(0)`). Остальные узлы наследуют диапазоны своих потребителей; когда потребители расходятся, `merge_consumer_ranges` либо сливает совместимые индексные выражения (части валидности объединяются через OR в `WHERE(valid, idx, Invalid)`), либо выделяет новые диапазоны и помечает ось для материализации. Операции перемещения отображают выходные диапазоны во входные через `apply_movement_op` (`PERMUTE` переставляет их, `EXPAND` обнуляет ось broadcast, `PAD` оборачивает диапазон в `WHERE` валидности, `RESHAPE` проходит через `apply_reshape_ranges`). `ending_ranges` распространяют решения о broadcast назад, так что `REDUCE`, питающий broadcast, материализуется до него (случай layernorm).
   - `apply_rangeify_patterns` (снизу вверх): `REDUCE` в тензорной форме → `REDUCE(src, ranges)` в форме цикла с `num_axes = 0`; `PAD` → `WHERE(valid, src, 0)`; `STACK` с формой → цепочка `WHERE` по его ведущему диапазону; материализуемые источники каждой операции оборачиваются в `STAGE` + `INDEX` (`transform_sources_with_bufferize`); после этого операции перемещения удаляются. Буфероподобный источник (`BUFFER`, `PARAM`, `SLICE`, `AFTER`, ...) получает один `INDEX` в построчном порядке, если его форма статическая (`linearize_static_indices`); изображения и символьные формы сохраняют по индексу на координату.
6. **Мега-проход** — одна неподвижная точка поверх `symbolic + pm_reduce_simplify + movement_op_patterns + buffer_folding + dead_axis_removal + pm_remove_bufferize`. Группы питают друг друга: встраивание `STAGE` открывает арифметику диапазонов, которую сворачивает `symbolic`, а это может сделать редукцию схлопываемой. Отдельные правила описаны на странице [оптимизации диапазонов](../optimizations/range-optimization.md).
7. **Пересборка SINK** по помеченному обратному срезу: источниками sink остаются только узлы `STAGE`, `MSTACK`, `CONST`, `PARAM` и `AFTER`, несущие выходной тег, в исходном порядке выходов.
8. **Лимит буферов** — если устройство сообщает `max_buffers`, `buffer_limit_patterns` принудительно переносит поэлементные источники в глобальные `STAGE`, чтобы ни одно ядро не превысило лимит аргументов.

Результат для `x.sum(1)` на тензоре `[8, 64]`:

```text
[67] SINK : Scalar(Void)
└── [66] STAGE : Scalar(Float32) shape=[Const(8)]
    ├── [65] CONTIGUOUS : Scalar(Float32) shape=[]
    │   └── [64] REDUCE(Add, num_axes=0, ranges=[27]) : Scalar(Float32) shape=[]
    │       ├── [62] INDEX : Scalar(Float32) shape=[]
    │       │   ├── [11] PARAM(slot=0) : Scalar(Float32) shape=[Const(512)]
    │       │   │   └── [0] CONST(Int(512)) : Scalar(WeakInt) shape=[]
    │       │   └── [55] Add : Scalar(WeakInt) shape=[]
    │       │       ├── [54] Mul : Scalar(WeakInt) shape=[]
    │       │       │   ├── [26] RANGE(U0, Weak) : Scalar(WeakInt) shape=[]
    │       │       │   │   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
    │       │       │   └── [3] CONST(Int(64)) : Scalar(WeakInt) shape=[]
    │       │       └── [27] RANGE(U1, Reduce) : Scalar(WeakInt) shape=[]
    │       │           └── [3] → (see above)
    │       └── [27] → (see above)
    └── [26] → (see above)
```

`U0`/`U1` — это `AxisId::Unrenumbered`: диапазоны перенумеровываются для каждого ядра при разрезе. `PERMUTE`/`RESHAPE` входа исчезли — они превратились в индексное выражение `U0 * 64 + U1`.

## Разрез на ядра (`try_get_kernel_graph`)

Сначала `kernel_graph_pre_cut`:

- **`pm_add_buffers_patterns`** (снизу вверх, `RangeifyBufferContext`): `movement_op_patterns`, затем `flatten_bufferize` (`STAGE` с несколькими диапазонами становится одним плоским диапазоном плюс обратный `RESHAPE`), `late_buffer_slice` (DISK `STAGE(BITCAST|CONTIGUOUS)` становится `SLICE`) и `bufferize_to_store`. Последний выделяет локальный для расписания `BUFFER` (`new_lunique_buffer`, слот в пространстве имён старших битов) и переписывает `STAGE(compute, ranges)` в `AFTER(BUFFER, [END(STORE(INDEX(BUFFER, idx), compute), ranges)])`. `STAGE(AFTER(..))` переиспользует нижележащий буфер; `STAGE` типа `Local` остаётся нетронутым до `pm_add_local_buffers`. Уже сформированный `SINK` ядра (с `KernelInfo`) закрыт гейтом, чтобы перезапись в него не спускалась.
- **`pm_flatten_range`** один раз по всему графу (снизу вверх): заново выводит список диапазонов каждого `END`/`REDUCE` из `RANGE`, достижимых через его источники, чтобы проход по ядрам ниже не обходил общие подграфы повторно.

Затем **`split_all_stores`** (снизу вверх): каждый `STORE` или `END(STORE)`, у которого не осталось открытых вычислительных диапазонов, становится `CALL`. `split_store` запускает `local_to_param_patterns + rangeify_codegen_patterns` на теле ядра: глобальные `BUFFER`/`PARAM` → `PARAM(slot)` кодогенерации, нумеруемые в порядке сопоставления через `LocalAddBufferContext::param_slot`; `BIND(var, value)` → переменная, а привязка сохраняется как аргумент `CALL`; `AFTER`/`MSTACK`/`MSELECT` → их буфер; `RANGE(end=0)` → `CONST(0)`; идентификаторы осей `Unrenumbered` → `Renumbered(n)`; `NOOP` → типизированный ноль; `CONTIGUOUS` → его источник, с извлечением подсказок. Тело оборачивается в `SINK` с `KernelInfo` по умолчанию; значение `COPY`/`SLICE` остаётся непосредственным телом вызова. Диапазоны `Device` — единственное исключение из правила «нет открытых диапазонов»: это линии запуска, и они переживают границу.

Наконец, **`validate_normal_kernel_devices`** (одно устройство на ядро, не являющееся копированием) и **`fix_assign`**: когда ядро B читает буфер, в который пишет ядро A, `AFTER` ядра A добавляется в зависимости `AFTER` ядра B; цикл даёт `KernelSplitDependencyCycle`. При включённом `SVOD_SPEC` результат проверяет `verify_kernel_graph`.

## Предоптимизация ядра (`apply_pre_optimization`)

Выполняется на теле каждого ядра до эвристик или BEAM, в обоих путях (`optimize_kernel_with_config_impl`, `optimize_kernel_beam`, `prepare_scheduler`). При включённом `SVOD_SPEC` сначала выполняется `type_verify` относительно `spec_tensor`.

| Шаг | Матчер | Направление |
|------|---------|-----------|
| операции перемещения | `movement_op_patterns` | снизу вверх |
| схлопывание загрузок | `pm_load_collapse` | сверху вниз |
| разбиение диапазонов | `pm_split_ranges + pm_flatten_range` (`SplitRangesContext`) | сверху вниз |
| символьное упрощение | `sym + pm_fold_cast_const + pm_flatten_range` | сверху вниз |
| упрощение диапазонов | `pm_flatten_range + pm_simplify_ranges` (`SimplifyRangesContext`) | сверху вниз |

**`movement_op_patterns`** содержит три правила: `INDEX(mop(x), idx)` → `INDEX(x, mop⁻¹(idx))` (`transform_movement_through_index`), `AFTER(mop(x) | INDEX(x), deps)` → `mop(AFTER(x, deps))` (`push_op_through_after`) и `END(mop(x), ranges)` → `END(x, ranges)`. `is_movement()` — это ровно `RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP`. Применяется снизу вверх, потому что внутренняя операция перемещения должна быть переписана раньше, чем её потребитель сможет сопоставиться.

**`pm_load_collapse`** устраняет `REDUCE(Add)`, тело которого после символьных рассуждений не зависит от диапазона (`reduce_load_collapse`): узлы вне области редукции заменяются скалярными переменными `PARAM` (`UOp::variable("in{n}", vmin, vmax)`), тело оборачивается в синтетический `REDUCE` по одному диапазону, запускается `build_reduce_load_collapse_matcher`, и если какой-то `RANGE` выжил, подстановка откатывается. Используемые им паттерны границ описаны на странице [оптимизации диапазонов](../optimizations/range-optimization.md).

**`pm_split_ranges`** записывает каждый `RANGE % const`, чей конец делится на константу (диапазоны `Warp` и `Device` исключены; каждый диапазон, которым индексируется `STORE` изображения, закреплён), и подставляет `r → outer * c + inner` один раз, на `SINK`, с идентификаторами осей `r.child(0)` / `r.child(1)`. Затем граф после подстановки упрощается через `symbolic + pm_fold_cast_const`.

**`sym`** — полный упроститель третьего уровня ([алгебраическое упрощение](../optimizations/algebraic-simplification.md)); `pm_fold_cast_const` сворачивает `CAST(CONST)`; `pm_flatten_range` поддерживает списки диапазонов корректными после исчезновения диапазонов.

**`pm_simplify_ranges`** сливает соседние диапазоны `END`/`REDUCE`, когда слитая форма не увеличивает число `FloorDiv`/`FloorMod` (`simplify_merge_adjacent`), и сужает диапазон до наибольшей границы, которую доказывает для него какой-либо гейт `INDEX` (`mark_gated`; одно использование без гейта закрепляет исходный конец; диапазоны `REDUCE` защищены). Обе подстановки происходят на `SINK`.

## Передача оптимизатору

`Scheduler::new(ast, renderer)` собирает `RANGE` с протяжённостью больше 1, отсортированные по `(axis_type.priority(), axis_id)`; `convert_loop_to_global` превращает выходные оси `Weak` в `Global` на рендерерах с `has_local` (на CPU это ничего не делает, поэтому ось строк в примере выше остаётся `Weak`). Затем `hand_coded_optimizations` или BEAM применяет `Opt`, и `get_optimized_ast_with_naming` выдаёт `SINK` ядра с метаданными `KernelInfo` (имя вроде `r_8_16_4`, `dont_use_locals`, `opts_to_apply`). `SVOD_NOOPT` пропускает эвристики, но не шаги этой страницы и не post-оптимизационные стадии. Сам поиск описан в разделе [поиск ядер](../optimizations/kernel-search.md).
