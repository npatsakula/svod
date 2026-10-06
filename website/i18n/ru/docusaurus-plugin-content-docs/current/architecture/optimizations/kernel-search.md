---
sidebar_label: Поиск ядер
---

# Поиск ядер: эвристики, BEAM и тензорные ядра

После `apply_pre_optimization` ядро — это гнездо циклов из диапазонов `Weak` и `Reduce`. Оптимизатор решает, как эти циклы исполняются — какие становятся измерениями сетки, рабочими группами, warp, векторными линиями, развёрнутыми телами или фрагментами тензорных ядер, — применяя `Opt` к `Scheduler`. Опции выбирают две стратегии: написанные вручную эвристики (по умолчанию) и поиск BEAM. Исходники: `schedule/src/optimizer/{scheduler,opts,heuristics,beam,tc,renderer,config}.rs`, `ir/src/opt.rs`.

## Планировщик и пространство действий

`Scheduler::new(ast, renderer)` индексирует `RANGE` ядра (протяжённость > 1), отсортированные по `(axis_type.priority(), axis_id)`; `convert_loop_to_global` превращает оси `Weak`, встречающиеся в каждом `STORE`, в `Global`, если у рендерера есть `has_local` (GPU), и ничего не делает на CPU. Затем `apply_opt(scheduler, opt, append)` переписывает по одному диапазону за вызов:

| `OptOps` | Эффект | Ограничения (`opts.rs`) |
|----------|--------|--------------------|
| `UPCAST(axis, n)` | отделить `n` линий от оси `Global`/`Local`/`Weak` как `Upcast` | `n <= renderer.upcast_max`; `n = 0` берёт всю ось |
| `UNROLL(axis, n)` | отделить `n` итераций от оси `Reduce`/`GroupReduce` как `Unroll` | `axis` индексирует `unrollable_dims()`; `n <= 32` |
| `LOCAL(axis, n)` | отделить измерение рабочей группы от оси `Global`/`Weak` | `has_local`, ранее не было `NOLOCALS` |
| `GROUP(axis, n)` / `GROUPTOP(axis, n)` | внутреннее / внешнее разбиение оси `Reduce` в `GroupReduce` (двухстадийная редукция через разделяемую память) | `has_local && has_shared`, укладывается в `shared_max`, не вложена в другую редукцию, **отвергается, если уже применена TC-опция** |
| `THREAD(axis, n)` | измерение ядер CPU на оси `Global`/`Weak`, которую можно сделать глобальной | `has_threads`, ещё нет оси `Thread`, `n <= global_max[0]` |
| `SWAP(a, b)` | поменять местами две оси `Global` | обе `Global` — поэтому никогда на CPU, где оси остаются `Weak` |
| `PADTO(axis, n)` | дополнить ось до кратного `n`, маскируя хвост | константная протяжённость, не `Upcast`/`Unroll`/`Thread`, дополнение меньше 4× объёма работы, `INDEX` с одним индексом |
| `NOLOCALS` | установить `dont_use_locals`, блокируя последующие `LOCAL`; gpudims запускает только глобальные измерения | ещё нет осей `Local`/`Warp`/`GroupReduce` |
| `TC(axis_choice, tc_select, tc_opt, use_tc)` | отобразить matmul на тензорное ядро | должна быть первой опцией; см. ниже |

`get_optimized_ast_with_naming` сплющивает списки диапазонов и прикрепляет `KernelInfo { name, applied_opts, dont_use_locals }`; имя — это `r_`/`E_` плюс протяжённости в порядке диапазонов (`r_8_16_4` в [разобранном примере](../codegen/worked-example.md)).

## Эвристики (`hand_coded_optimizations`)

`hand_coded_optimizations(&mut scheduler, &HeuristicsConfig)` применяет в таком порядке (`heuristics.rs`):

1. **`try_tensor_cores`** — если `tc_enabled != Disabled`, у рендерера есть тензорные ядра и (при `TcOpt::Strict`) ровно одна ось редукции: `tc::detect_matmul`, затем `apply_with_axis_choice` по вариантам выбора осей, затем `apply_tc_tiling` — `FixedStep`: `UPCAST` M и N на первое из 5/4/3/2, которое делит, `LOCAL` N на 4 или 2; `LaneBudget { accum_max: 128 }` (CUDA sm75/80/89): `tc_warp_tile_growth`, затем `LOCAL` размером `wave_size / tc.threads`. При успехе возвращается.
2. **`apply_image_upcasts`** — буферы изображений.
3. **`apply_matvec_fast_path`** — конфигурация matvec `SVOD_MV*` (`PADTO`, `UPCAST` малых осей, `GROUP` по возможности, `LOCAL`, `UPCAST`, `UNROLL`). При успехе возвращается.
4. **`try_grouped_reduction`** — `GROUPTOP(axis, 16)` для выхода не более чем из 2048 элементов (240 без локальных измерений); иначе **`try_warp_row_reduction`** (`GROUP` на размер wave плюс `UNROLL 4`). Если теперь существует ось `GroupReduce`, функция возвращается.
5. **`apply_masked_upcasts`** — маскированные оси размера 2–7 с произведением ≤ 49.
6. **`apply_heuristic_upcasts`** — пока у выхода ≥ 1024 элементов и произведение upcast меньше 32, `UPCAST` на 3 или 4; оси ранжируются по `(num_strides, sum_strides, axis, vector rank)`.
7. **`apply_unroll`** — ось редукции разворачивается полностью, если ≤ 32 (вторая тоже, если обе ≤ 3), иначе `UNROLL 4`.
8. **`apply_default_upcast`** — `UPCAST 4` на последней оси, допускающей upcast, если ещё ничего не было подвергнуто upcast или unroll.
9. **`apply_local_dims`** — размеры `LOCAL` `[32, 16, 8, 4, 3, 2]` для оси 0 и `[16, 8, 4, 3, 2]` для остальных, совокупный бюджет 128, не более трёх, с запасным вариантом через `PADTO`.
10. **`apply_threading`** — только CPU: `THREAD` на `[32, 16, 12, 8, 6, 5, 4, 3, 2]` по оси `Weak`, оставляя не менее 131072 элементов на поток, с запасным вариантом `PADTO` + `THREAD`.

`HeuristicsConfig::from_env` читает `SVOD_TC` (0 — выключено, 2 — только формы, иначе включено), `SVOD_TC_OPT`/`TC_OPT`, `SVOD_TC_SELECT`/`TC_SELECT`, `SVOD_MV*`, `SVOD_NOLOCALS`, `SVOD_THREADS`. Пороги групповой редукции — константы в `heuristics.rs`; `SVOD_K_VECTORIZE` и `SVOD_NO_OUTPUT_UPCAST` устанавливают поля, которые на этом пути никто не читает.

## Поиск BEAM

`BEAM=N` (N > 0) выбирает `OptStrategy::Beam { width: N }`. Тогда `realize` направляет ядро через `beam_search_cached_remote(scheduler, config, compiler_identity, behavior_fingerprint, compile_wave, benchmark)` (`beam.rs`); у простого API `optimize_kernel_with_config` нет замыкания для компиляции и замера времени, и он откатывается к эвристикам.

Поиск (`beam_search_remote_staged`):

1. Начать с `[(scheduler, Duration::MAX)]` и добавить результат эвристик как дополнительного кандидата первой волны.
2. **Расширение**: для каждого члена луча `generate_actions` пробует каждое из 193 действий `BEAM_ACTIONS` (200 с `BEAM_PADTO`): `passes_prefilter` (ось существует; действие, чья величина равна размеру оси, пропускается, если есть вариант с `0`), `apply_opt`, `validate_limits` (`upcast_prod / tc_up <= max_upcast`, `local_prod <= max_local`). `NOLOCALS` добавляется для каждого члена при `enable_nolocals`.
3. **Компиляция** кандидатов в пуле рабочих процессов; кандидат отбрасывается там, если число операций после линеаризации достигает `max_uops` или компиляция превышает `compile_timeout_secs`.
4. **Фильтрация**: отбрасываются кандидаты, у которых `compute_ops` больше чем в 1000× превышает минимальное в волне, затем дубликаты по ключу бинарника (или исходника).
5. **Замер времени**: по `num_runs` запусков на каждого, оценка = минимум; запуск прерывается при 3× от текущего лидера; глобальный размер ограничивается 65536, а время масштабируется обратно.
6. **Сохранение** лучших `beam_width`. Остановка, когда лучшее время больше не улучшается на `min_progress_ns` (или уже меньше этого значения); если улучшение было, луч схлопывается до единственного победителя для следующей волны.

Список действий (`BEAM_ACTIONS`): `UPCAST` с величинами `[0,2,3,4,5,7]` × оси 0..8 (48), `UNROLL` `[0,4,7]` × 0..5 (15), `LOCAL` `[2,3,4,8,13,16,29]` × 0..6 (42) плюс `(0,32)` и `(6,2)`, `GROUPTOP` `[13,16,28,29,32,49,64,256]` × 0..3 (24), `GROUP` `[0,4,8,16]` × 0..3 (12), `TC` (одно действие `tc_opt = 0` плюс девять вариантов выбора осей при `TC_OPT`), пары `SWAP` в пределах 0..5 (10), `THREAD` `[2,3,4,5,8,12,16,24,32,64]` × 0..3 (30). `BEAM_PADTO` добавляет `PADTO(axis, 32)` для осей 0..7.

### Кеш

Результаты сохраняются в базе `sled` в `$SVOD_BEAM_CACHE_DIR/beam_cache`, иначе в `~/.cache/svod/beam_cache` (`dirs::cache_dir()`). Ключ (`CacheKey`, схема 11) — структурный хеш AST плюс ширина луча, устройство, `renderer.cache_fingerprint()`, идентичность компилятора, лимиты (`max_upcast`, `max_local`, `max_uops`, `num_runs`, `min_progress_ns`, `enable_nolocals`, `compile_timeout_secs`), отпечаток поведения (`transcendental`, `disable_fast_idiv`) и хеш пространства действий. Значение — список `applied_opts`; попадание воспроизводится через `replay_opts`, проверяется и замеряется один раз и инвалидируется, если это не удалось. `IGNORE_BEAM_CACHE=1` обходит кеш, `clear_cache` очищает его.

### Переменные окружения

| Переменная | По умолчанию | Значение |
|----------|---------|---------|
| `BEAM` | 0 | ширина луча; 0 = эвристики |
| `BEAM_UPCAST_MAX`, `BEAM_LOCAL_MAX`, `BEAM_UOPS_MAX` | 256, 1024, 3000 | `validate_limits` и лимит операций в рабочем процессе |
| `BEAM_RUNS` | 3 | число замеров на кандидата |
| `BEAM_MIN_PROGRESS` | 10 (мкс, хранится в нс) | порог остановки |
| `BEAM_PADTO` | 0 | добавить семь действий `PADTO` |
| `NOLOCALS` / `SVOD_NOLOCALS` | не задана | добавить действие `NOLOCALS` |
| `PARALLEL` | 0 | рабочие процессы компиляции (на GPU по умолчанию — бюджет потоков, иначе 1) |
| `BEAM_TIMEOUT_SEC`, `BEAM_MAX_TASKS_PER_CHILD` | 10, 16 | сторожевой таймер и перезапуск рабочих процессов |
| `TC`, `TC_OPT` | 1, 2 | действия BEAM для тензорных ядер (`TC_SELECT` под BEAM игнорируется: всегда `Auto`) |
| `BEAM_DEBUG`, `BEAM_LOG_SURPASS_MAX` | не задана | диагностика |
| `IGNORE_BEAM_CACHE`, `SVOD_BEAM_CACHE_DIR` | не задана | управление кешем |

:::tip[BEAM не читает переключатели эвристик]
Эвристическое начальное значение внутри BEAM использует `HeuristicsConfig::from_env()`, но собственные TC-действия поиска читают `TC` и `TC_OPT`, а не `SVOD_TC`/`SVOD_TC_OPT`. `SVOD_NOOPT` (с любым значением) выбирает `OptStrategy::None`: никаких опций, но пред- и post-оптимизация по-прежнему выполняются.
:::

## Тензорные ядра

`renderer.rs` хранит таблицу тензорных ядер для каждого `RendererDevice`; измерения — `(N, M, K)`:

| Цель | Ядра (вход → выход) | потоки |
|--------|------------------|---------|
| CUDA sm75 | 8×16×8 f16→f32, f16→f16 | 32 |
| CUDA sm80 | 8×16×16 f16→f32, bf16→f32, f16→f16; 8×16×8 f16→f32, f16→f16; 8×16×32 i8→i32; опционально tf32 8×16×8 | 32 |
| CUDA sm89 | sm80 плюс 8×16×32 fp8 e4m3/e5m2→f32 | 32 |
| AMD RDNA3 | 16×16×16 f16→f32, f16→f16, bf16→f32, i8→i32 | 32 |
| AMD RDNA4 | RDNA3 плюс bf16→bf16 | 32 |
| AMD CDNA3 | 16×16×32 fp8 e5m2/e4m3; 16×16×16 f16/bf16→f32 | 64 |
| AMD CDNA4 | CDNA3 плюс 16×16×128 fp8 | 64 |
| Metal | варианты 8×8×8 f32/f16/bf16 | 32 |
| Intel Xe | 8×8×16 f16→f32 | 8 |
| WebGPU, CPU | нет | — |

`for_cuda_arch` выбирает профиль sm80, если у вычислительной возможности (capability) есть bf16 mma, иначе sm75, и не даёт тензорных ядер ниже sm75.

`tc.rs`: `detect_matmul` находит `REDUCE(Add, MUL(in0, in1), reduce_ranges)`; диапазоны, которые использует только `in0`, — кандидаты в M, те, что использует только `in1`, — в N, диапазоны редукции — K, и каждая тройка `(M, N, K)` — вариант выбора осей (диапазон M/N, который сам является осью `Reduce`, отвергается); `select_tensor_core` сопоставляет скалярные dtype входа и выхода (вход fp8 без нативного ядра откатывается на ядро f16); `apply_with_axis_choice` перебирает варианты выбора осей × ядра в пределах бюджета из 64 попыток. Ядро применяется разбиением осей: один диапазон `Warp` протяжённостью `tc.threads`, ось `Upcast` размера 2 на каждую запись `TcOpt::Upcast`, каждая запись `TcOpt::Local` берёт цифру индекса warp (`warp % 2`, `warp / 2`), K становится `log2(K)` осями `Unroll` размера 2; оставшиеся N/M остаются `Global`, а оставшиеся оси редукции оборачивают `WMMA` в `REDUCE`. `TcUsage::ShapeOnly` (`SVOD_TC=2`) выполняет разбиения, но не выдаёт `WMMA`.

Уровни `TcOpt` (`TC_OPT`): **0 Strict** — только одна ось редукции, M/N/K должны делиться; **1 Relaxed** — то же правило делимости внутри `tc.rs`; **2 Padded** (по умолчанию) — `PADTO` неделящейся оси, если дополнение добавляет не более 25%; **3 Unbounded** — дополнение в пределах собственного лимита 4× у `PADTO`. Символьная ось никогда не использует тензорное ядро.

## Программная конфигурация

```rust
use svod_schedule::optimizer::{OptStrategy, OptimizerConfig};
use svod_tensor::PrepareConfig;

let config = PrepareConfig::from(
    OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 8 })
        .build(),
);
tensor.realize_with(&config)?;
```

`OptimizerConfig` (builder на `bon`) имеет поля `strategy`, `beam: BeamConfig`, `heuristics: HeuristicsConfig`, `transcendental` (`TRANSCENDENTAL`, по умолчанию 1; ≥ 2 принудительно включает полиномиальные декомпозиции), `disable_fast_idiv` (`DISABLE_FAST_IDIV`, по умолчанию **1**: деление через магическое число включается через `DISABLE_FAST_IDIV=0`) и `opts_to_apply` (явный список опций, который также можно прочитать из `KernelInfo` у `SINK` ядра; он переопределяет стратегию, а опция, которую не удалось применить, — ошибка). `PrepareConfig` имеет `optimizer`, `planner_mode`, `disable_schedule_cache`, `device_local_outputs`, `threads` и конструкторы `Default`, `from_env`, `device_local`, `for_cpu_backend`, `for_{amd,metal,cuda}_if_available`.
