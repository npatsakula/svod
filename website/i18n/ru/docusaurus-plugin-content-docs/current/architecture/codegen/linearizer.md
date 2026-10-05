---
sidebar_label: Поздние перезаписи и линеаризатор
---

# Поздние перезаписи, граница программы и линеаризатор (стадии 19–20 и далее)

Последние post-оптимизационные стадии делают граф пригодным для рендеринга под один конкретный бэкенд: операции, которых нет у цели, раскладываются, валидность становится `gate`, слабые dtype фиксируются. Затем `svod-codegen` добавляет рёбра потока управления, нумерует параметры и сплющивает DAG в список инструкций. Исходники: `optimizer/mod.rs`, `late/gater.rs`, `late/dtype.rs`, `optimizer/implicit_barriers.rs`, `linearize/`, `codegen/src/program_pipeline.rs`.

Каждый матчер ниже строится из таблицы возможностей рендерера (`renderer.supported_ops()`, `supports_dtype`), поэтому `optimize_kernel_with_config` отказывается работать с рендерером без неё (`OptError::MissingRendererCapabilities`).

## 19 — приведение операндов float ALU

`pm_cast_float_alu`: для `Sin`, `Log2`, `Exp2`, `Sqrt`, `Reciprocal` приводит операнд к dtype результата. Декомпозиции трансцендентных функций разворачиваются в однородные по dtype полиномы и не должны видеть операнд смешанного dtype.

## 19b — ранние декомпозиции

`early_decomposition_patterns(supported_ops)`:

```text
symbolic_simple + pm_fold_cast_const + pm_mod_to_and + divmod_decomposition_patterns
  + pm_threefry_decomp        if !supports(Threefry)
  + pm_max_decomposition      if !supports(Max) && supports(Lt)
  + pm_erf_decomposition      if !supports(Erf)
```

`divmod_decomposition_patterns` (`ir/src/decompositions/mod.rs`) понижает деление и остаток с округлением вниз (`FloorDiv`/`FloorMod`) до усекающих `CDiv`/`CMod` с коррекцией знака — формы, которая есть у любого бэкенда. `pm_mod_to_and` стоит здесь, как и в позднем наборе, чтобы остатки по степени двойки сворачивались до того, как их увидит усекающее понижение.

## 19c — декомпозиции dtype

`pm_dtype_decomp_commit = pm_dtype_decomps + pm_commit_weak` с `DTypeDecompCtx`. Первое правило лишь записывает, какие из `FP8E4M3`, `FP8E4M3FNUZ`, `FP8E5M2`, `FP8E5M2FNUZ`, `Float16`, `BFloat16`, `Int64`/`UInt64` встречаются в графе; затем правило `SINK` переписывает снизу вверх и в порядке dtype каждый записанный dtype, который рендерер не поддерживает:

| Не поддерживается | Эмулируется как | Матчер |
|-------------|-------------|---------|
| `Int64`, `UInt64` | два слова `Int32`/`UInt32`: `PARAM`/`BUFFER` удвоенного размера, `INDEX` помечен словом, переносы и заёмы строятся из `Lt`, 64-шаговый делитель сдвигом и вычитанием для `CDiv`/`CMod` | `pm_long_decomp` (`devectorize.rs`) |
| FP8 | `Float16`, если поддерживается, иначе `Float32`; хранение остаётся 8-битным беззнаковым словом, `f2f` выполняет побитово точное преобразование (округление RNE, кодирование NaN для FNUZ, насыщение в `f2f_clamp`) | `pm_float_decomp` |
| `Float16`, `BFloat16` | вычисления в `Float32`, то же преобразование хранения через `f2f` | `pm_float_decomp` |

`get_dtype_decomps` возвращает тот же выбор в виде списка для ключа кеша компиляции рендерера.

## 19d — поздние декомпозиции

`pm_decomp = early_decomposition_patterns + get_late_rewrite_patterns(renderer, disable_fast_idiv) + get_transcendental_patterns(supported_ops, TRANSCENDENTAL >= 2) (+ renderer.decomposition_matcher())`, выполняется до неподвижной точки. Поздний набор зависит от возможностей рендерера (все правила — на странице [снижения стоимости операций](../optimizations/strength-reduction.md)):

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

`get_transcendental_patterns` (`ir/src/decompositions/`) заменяет `Exp2`, `Log2`, `Sin` полиномиальными приближениями `xexp2`/`xlog2`/`xsin` для f16/f32/f64 (остальные float вычисляются в f32), а `Sqrt` — на `xpow(x, 0.5)`, для каждой операции, которой нет у рендерера, либо для всех сразу при `TRANSCENDENTAL=2`. `decomposition_matcher` — копия на стороне оптимизатора хука устройства `Renderer::decompositor()`; Metal устанавливает туда `amd_decomposition_patterns` (`Exp`, `Log`, `Cos`, `Tan` и бинарный `Pow` поверх нативных `exp2`/`log2`).

В разобранном примере эта стадия превращает `R1 * 64` в `R1 << 6`, а `R0 * 4 + (R1 << 6)` — в `MulAcc(R0, 4, R1 << 6)`, целочисленный FMA, построенный `pm_shl_add_to_mulacc`.

## 19e — гейты, регистровые линии, понижение float

`pm_move_gates_from_index` (`late/gater.rs`, порт `gater.py` из Tinygrad) наконец выносит валидность из индекса:

| До | После |
|--------|-------|
| `LOAD(INDEX(buf, WHERE(g, idx, Invalid)))` (без `alt`, без `gate`) | `LOAD { index: INDEX(buf, idx), alt: 0, gate: g }` |
| `STORE(INDEX(buf, WHERE(g, idx, Invalid)), v)` (без `gate`) | `STORE { index: INDEX(buf, idx), value: v, gate: g }` |
| те же две формы на `SHRINK` (слитые группы) | `LOAD`/`STORE` с гейтом на очищенном `SHRINK` |
| двухкоординатный `INDEX` изображения с одним общим условием | один доступ с гейтом (проверяется первым) |
| `WHERE(g, LOAD{gate: g}, alt)` и инвертированная форма | `alt` сворачивается в загрузку |

`valid_index` требует буквальную константу `Invalid` в третьем слоте `WHERE`. Затем `pm_scalarize_register_stack_index_preserve_deps` разрешает `INDEX(AFTER(STACK(..), deps), c)` — линию регистрового стека, прочитанную после некоторых записей, — в выбранный `LOAD` с зависимостями, повторно прикреплёнными к его адресу, а `merge_register_read_ends` сливает `END`, закрывающие одни и те же диапазоны, под одним регистровым `AFTER` (отладочная проверка убеждается, что не выжил ни один `INDEX` регистрового стека). Последним выполняется `demote_unsupported_floats` (`late/dtype.rs`): на рендерере без ALU `Float64` (Metal, WebGPU) каждое внутреннее значение f64 вычисляется в f32, тогда как глобальное хранение f64, его загрузки и их значения `alt` сохраняют широкий dtype.

## 20 — финальная перезапись

```text
pm_final = pm_commit_weak + pm_cast_weak + pm_decomp (+ renderer.extra_matcher()) + pm_split_ends
```

Одна неподвижная точка (в отладочных сборках сначала выполняется `assert_target_renderer_boundary`: нет статического многоиндексного `INDEX`, нет остаточного синглтонного broadcast, нет смешанного `STACK`/векторного ALU). `pm_split_ends` превращает `END(x, [r1, r2, r3])` в `END(END(END(x, r3), r2), r1)`, диапазоны отсортированы по убыванию `(axis_id, axis_type.priority())`; источники `Void`/`Bool` (обратные рёбра редукций) выделяются и снова прикрепляются к самому внешнему `END`, а исходный тег сохраняется, чтобы последующие шаги слияния его нашли. `extra_matcher` — хук конкретного бэкенда на `svod_device::device::Renderer`; он выполняется в той же неподвижной точке, что и декомпозиции. Рендереры CPU и NVPTX снова устанавливают `bool_storage_patterns` (`cpu_extra_matcher`), AMD устанавливает `amd_non_native_fp8_patterns` (ALU OCP FP8 расширяется до f32; хранение, преобразования и операнды MFMA не затрагиваются).

Затем, отдельными проходами: `pm_remove_invalid` заменяет каждый оставшийся `WHERE(c, x, Invalid)` с типом данных на `WHERE(c, x, 0)`, а каждую линию `STACK` со значением `Invalid` — на ноль (отладочная проверка убеждается, что ничего не осталось), и `add_implicit_barriers` вставляет `BARRIER` для локальной памяти: RAW-барьер перед `AFTER` на локальном буфере, чьи зависимости содержат локальный `STORE` без барьера, и WAR-барьер в конце тела цикла, который пишет в локальный буфер, читаемый другой загрузкой в том же цикле. `optimize_kernel_with_config_and_final_rewrite` возвращает граф, захваченный непосредственно перед барьерами, для инструментов паритета. Метаданные `KernelInfo`, отброшенные `graph_rewrite`, прикрепляются заново.

## Граница программы (`program_from_sink`)

Управление переходит к `svod-codegen` (`program_pipeline.rs`):

1. **`add_control_flow`** (`linearize/mod.rs`): снова `pm_split_ends` (идемпотентен), затем `CFGContext::new(sink)` и `pm_add_control_flow` снизу вверх. Контекст вычисляет для каждого `END`, в какой `END`/`SINK` он вложен — `END x` вложен в `u`, когда `u` зависит от `x`, а диапазон `u` входит в зависимости `x`, — группирует соседей по родителю, упорядочивает их по числу соседей, от которых они зависят, и записывает ребро от `RANGE` каждого последующего соседа к его предшественнику (`END` предыдущего соседа или `RANGE` родителя для первого). `pm_add_control_flow` добавляет предшественника в источники `RANGE`; после этого `InScopeRangesProperty` видит вложенность, и именно это даёт вложенным диапазонам больший `run_count` ниже. Предшественник, который уже содержит диапазон, вызывает панику (`"edge would create cycle"`).
2. **`number_params`** назначает финальные слоты `PARAM` (`validate_param_slots` отклоняет неназначенный или дублирующийся слот).
3. **`verify_final_sink`** относительно `spec_program` при включённом `SVOD_SPEC`; `ProgramInfo::from_sink` читает ABI.
4. **`pre_isel_matcher` / `isel_matcher`** — два хука выбора инструкций на `svod_device::device::Renderer`, оба снизу вверх (`PreIselContext`, `IselContext`). Они существуют для бэкендов уровня ISA; рендереры LLVM и C оставляют их `None`.
5. `UOp::program(sink, info, None, None, None)` — узел `PROGRAM`, чьи последующие источники — стадии `LINEAR`, `SOURCE` и `ProgramBinary`.

## `linearize`

`do_linearize` вызывает `svod_schedule::linearize(sink)` (`linearize/linearize.rs`, прямой порт `linearizer.py` из Tinygrad), затем `line_rewrite_cleanups`, затем `verify_linear_list` относительно `spec_program`.

Ключ сортировки каждого узла — `(run_count, priority, extra, tuplize rank)`:

| Операция | Приоритет |
|----|----------|
| `PARAM` | −20, при равенстве решает слот (`extra`) |
| `BUFFER` (глобальный, регистровый) | −18 |
| `BUFFER` (`AddrSpace::Local`) | −17 |
| `END` | −5 |
| `LOAD` | −1 |
| всё остальное (`CONST`, ALU, `SPECIAL`, …) | 0 |
| `STORE` | +1 |
| `RANGE` | +5 |

`run_count = prod(vmax + 1)` по находящимся в области видимости диапазонам узла (символьная протяжённость считается за 1), поэтому код вне цикла сортируется раньше тела цикла. Ранг tuplize — это ключ `(op, arg, dtype, *src.tuplize)` из Tinygrad, вычисляемый итеративно по топологической сортировке (`TuplizeKeys`), что делает порядок полным и детерминированным. Затем линейный список строится топологической сортировкой с max-кучей от `SINK` по этим рангам и в конце разворачивается: узел выдаётся, когда выданы все его потребители, поэтому определения идут первыми, `LOAD` — перед своими использованиями, `STORE` — после вычисления, а каждый `RANGE` открывается непосредственно перед своим телом.

`SVOD_DUMP_LINEAR=<dir>` записывает топологическую сортировку с id находящихся в области видимости диапазонов (`tree_<id>.txt`) и итоговый список (`linear_<id>.txt`).

## `line_rewrite_cleanups`

`line_rewrite` проходит по списку инструкций один раз; каждая запись может быть заменена несколькими. Единственная очистка — `linearize_cleanup_pattern`: `STORE` с гейтом `Bool`, чей адрес — `INDEX`/`SHRINK` (возможно за `CAST`), становится `IF(gate)`, `STORE` без гейта, `ENDIF`. И `IF`, и `ENDIF` существуют только в списке, но никогда в графе (`"if not allowed in graph"`), поэтому `spec_program` проверяется и на списке, и на sink. Бэкенды, умеющие предикатировать запись (LLVM, CUDA, Metal), рендерят эту тройку как условную запись.

Список — это стадия `LINEAR` у `PROGRAM`; `do_render` передаёт его в `Renderer::render`, а `do_compile` производит бинарник. [Разобранный пример](./worked-example.md) показывает LLVM IR, который CPU-рендерер выдаёт для ядра суммы по строкам.
