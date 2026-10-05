---
sidebar_label: Expander и редукции
---

# Expander и понижение редукций (стадии 08–11)

Первые четыре post-оптимизационные стадии берут выход оптимизатора — ядро, чьи `RANGE` теперь несут типы осей `Upcast`/`Unroll`/`Global`/`Local`/`GroupReduce`, — и делают намерение конкретным: развёрнутые диапазоны становятся константами с формой, `REDUCE` становится циклом с аккумулятором, локальные `STAGE` становятся локальными буферами. Все они выполняются внутри `apply_post_optimization_configured_with_capture` (`optimizer/mod.rs`).

## 08 — post-opt символьное упрощение

`POST_OPT_SYM = sym() + pm_move_where_on_load() + pm_flatten_range() + pm_reduce_unparented()`, одна неподвижная точка сверху вниз. Порядок в исходнике важен: более поздние группы потребляют то, что производят ранние.

- `sym()` — полный упроститель третьего уровня ([алгебраическое упрощение](../optimizations/algebraic-simplification.md)).
- `pm_move_where_on_load` (`symbolic/patterns.rs`) переписывает `WHERE(cond, INDEX(buf, idx), 0)` в `INDEX(buf, WHERE(cond', idx, Invalid))`. Условие разбивается по `AND`; клауза переезжает в индекс, только если все её диапазоны находятся в области видимости `INDEX` и у неё нет собственной зависимости от `INDEX`; остальные клаузы остаются во внешнем `WHERE`. Инвертированная форма `WHERE(cond, 0, INDEX(..))` обрабатывается с отрицанием условия. Теперь валидность едет внутри индексного выражения, где её видят девекторизатор и `indexing_simplify`; гейтом (`gate`) LOAD/STORE она становится только на `19e`.
- `pm_flatten_range` пересобирает списки диапазонов `END`/`REDUCE`.
- `pm_reduce_unparented` отбрасывает диапазоны редукции, на которые тело не ссылается: `Add` умножает на протяжённость, `Mul` возводит в степень протяжённости, `Max` просто отбрасывает диапазон (ветви `Min` нет; редукции `Min` не сопоставляются).

## 09 — expander (`pre_expand`)

`expander2() + pm_flatten_range() + mop_cleanup_patterns()` с контекстом `RangeMap` (`expand.rs`). `build_range_map` назначает каждому `RANGE` типа `Upcast`/`Unroll` координатную позицию в порядке топологической сортировки; длина карты — ранг значений с формой, которые создаёт эта стадия.

Три правила в порядке исходника:

| Правило | Эффект |
|------|--------|
| `Reduce { .. }` → `expand_reduce` | `REDUCE` в форме цикла, чей список диапазонов содержит элементы с формой, не являющиеся `RANGE`, превращает оси этих элементов (протяжённость > 1) в ведущие *горизонтальные* оси: источник переставляется так, чтобы они шли первыми, и `num_axes` их считает; результат переформируется, сохраняя заглушки размера 1. |
| `Range { axis_type: Upcast \| Unroll }` → `expand_range` | Диапазон становится `RESHAPE(STACK(CONST(0), ..., CONST(end-1)), shape)`, где `shape` состоит из единиц везде, кроме собственной координаты диапазона. Каждый потребитель диапазона получает форму через broadcast; пока ничего не дублируется. |
| `Wmma { metadata.upcast_axes: Some(..) }` → `expand_wmma` | `contract_axis` переносит координаты upcast для A/B в хвост и сплющивает их в операнды фрагментов; `unroll_axis` восстанавливает координаты C на выходе. Поле `upcast_axes` метаданных очищается. |

`mop_cleanup_patterns` (`devectorize.rs`) — это `mop_cleanup` из Tinygrad: слить вложенные `RESHAPE`, отбросить тождественные `RESHAPE`/`PERMUTE`, слить цепочки `PERMUTE`, схлопнуть `STACK(INDEX(b,0), INDEX(b,1), ..)` обратно в `b`, свернуть `INDEX(STACK(..), const)` в линию и скомпоновать `INDEX(INDEX(b, i), j)` в `INDEX(b, i, j)`, когда индексы скалярные. Символьный матчер здесь не запускается.

В разобранном примере диапазон редукции `R2` (`Unroll`, протяжённость 4) исчезает, а индекс получает форму:

```text
[151] REDUCE(Add, num_axes=1, ranges=[118]) : Scalar(Float32) shape=[]
├── [149] INDEX : Scalar(Float32) shape=[Const(4)]
│   ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
│   └── [148] Add : Scalar(WeakInt) shape=[Const(4)]
│       ├── [147] Add : Scalar(WeakInt) shape=[Const(4)]
│       │   ├── [119] Mul : Scalar(WeakInt) shape=[]          ← R0 * 4
│       │   └── [146] STACK(len=4) : Scalar(WeakInt) shape=[Const(4)]
│       └── [90] Mul : Scalar(WeakInt) shape=[]              ← R1 * 64
└── [118] RANGE(R0, Reduce)
```

`expand_reduce` уже превратил ось из 4 линий в `num_axes=1`, так что редукция по линиям горизонтальная, а оставшийся цикл идёт только по `R0`.

:::tip[STACK — единственная векторная операция]
Значение с формой — это `STACK` линий (возможно вложенный, возможно за `RESHAPE`). `INDEX(STACK(..), c)` выбирает линию той же операцией, что адресует буфер. Пары операций vectorize/contract нет, а `Upcast`/`Unroll` — это `AxisType`, а не операции.
:::

## 10 — понижение редукций (`pm_reduce`)

`movement_cleanup_patterns() + pm_reduce_local()` с `ReduceContext`. `movement_cleanup_patterns` — это `mop_cleanup_patterns` плюс два правила, нужных только девекторизатору (`RESHAPE(STACK([x]))` → `x`, когда формы совпадают; `RESHAPE`, который только добавляет ведущие измерения размера 1 → по одной обёртке `STACK([..])` на каждое добавленное измерение).

`pm_reduce_local` (`devectorize.rs`) компонует по порядку:

1. **`pm_wmma_add`** — `WMMA(a, b, c) + add` → `WMMA(a, b, c + add)`, также сквозь обёртки `PERMUTE` и `PERMUTE(RESHAPE(..))`, которые `expand_wmma` оставил на выходе. `try_add` отказывается при несовпадении dtype вместо assert.
2. **`pm_group_for_reduce`** (`expand.rs`) — `REDUCE` с диапазонами `GroupReduce` превращается в: частичный `REDUCE` по остальным диапазонам → `STAGE` частичного результата с находящимися в области видимости диапазонами `Local` плюс групповыми диапазонами (`BufferizeOpts::local_for_axis`) → `INDEX` этой стадии с локальными диапазонами и свежими циклами `Reduce` (`axis_id.group_reduce_loop()`) → финальный `REDUCE` по этим циклам.
3. **`reduce_to_acc`** — `REDUCE` с диапазонами. Если `num_axes > 0`, линии сначала сворачиваются слева направо в построчном порядке (`horizontal_reduce`). Затем:

   ```text
   acc        = BUFFER(slot, AddrSpace::Reg)                       // placeholder_like(red)
   acc_init   = STORE(AFTER(acc, input_ranges), identity)           // 0 for Add, 1 for Mul, dtype min/max for Max/Min
   acc_loop   = AFTER(acc, [acc_init, reduce_ranges..])
   body       = op(acc_loop, horizontal_inp)                        // Add/Mul/Max; float Min is -(max(-a, -b))
   store_end  = END(STORE(acc, body), reduce_ranges)   tag=TAG_MERGEABLE
   result     = AFTER(acc, [store_end])
   ```

   `input_ranges` — диапазоны, находящиеся в области видимости на входе, которые не редуцируются и ещё не закрыты, так что инициализация оказывается внутри охватывающих циклов. Конструкции цикла нет: `END` закрывает диапазоны редукции, а цепочка `AFTER` — это зависимость по данным.
4. **`expand_horizontal_reduce`** — `REDUCE`, у которого не осталось диапазонов, — это только свёртка линий.
5. **Слияние END** — на `SINK` функция `merge_reduce_ends` группирует `END` с `TAG_MERGEABLE` по множеству диапазонов редукции и контексту вложенности и заменяет каждую группу на `END(GROUP(computations), ranges)`; группы на другой глубине вложенности получают клонированные `RANGE` со свежими идентификаторами осей, чтобы каждый диапазон закрывался ровно одним `END`.
6. **`clean_up_group_sink`** — `GROUP` с одним источником разворачиваются; источники `NOOP`/`STACK`/`SINK`/`GROUP` у `SINK` или `GROUP` сплющиваются.

`Min` на числах с плавающей точкой понижается через `Max` (`-(max(-a, -b))`), чтобы NaN вёл себя как в редукции max; на целых это `WHERE(a < b, a, b)`.

## 11 — локальные буферы

`pm_add_local_buffers = { Stage => add_local_buffer } + movement_op_patterns` (`optimizer/mod.rs`). Каждый `STAGE`, доживший до этого места, только что создан `pm_group_for_reduce` (глобальные стали `STORE` при разрезе, а `bufferize_to_store` намеренно пропустил `Local`). `add_local_buffer` выделяет `UOp::placeholder(max_shape, dtype, slot, opts.addrspace)` — слот берётся из `LocalBufferContext::axis_slot` групповой оси, детерминированного хеша для путей вложенных осей — и переписывает стадию в `AFTER(buffer, [END(STORE(INDEX(buffer, ranges), compute), ranges)])`. Затем `movement_op_patterns` проталкивает в индексное выражение любую операцию перемещения, под которой находится новый `INDEX`.

Tinygrad понижает редукции до добавления локальных буферов по той же причине: стадия групповой редукции не существует, пока не выполнен шаг 2 `pm_reduce_local`.
