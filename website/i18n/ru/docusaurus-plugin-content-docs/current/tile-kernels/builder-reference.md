---
sidebar_label: API билдера
---

# API билдера

В главе [Написание ядра](./first-kernel) понадобилась лишь горстка вызовов. Эта страница —
полное лицо AUTHOR у `svod-tk`: все типы и методы, которыми пишется тело ядра, сгруппированные
по назначению. Всё перечисленное ре-экспортируется из `tk/src/lib.rs`, если не указан путь модуля.

Тело ядра — это замыкание `FnOnce(&Kernel) -> Arc<UOp>`. Оно привязывает глобальные буферы,
выделяет тайлы, выдаёт операции над тайлами через `Group` и возвращает `ker.finish(n)`.

---

## `Kernel` — контекст

```rust
// tk/src/kernel.rs
pub fn new(name: impl Into<String>, grid: [i64; 3], block: i64, buffers: Vec<Arc<UOp>>, caps: ArchCaps) -> Kernel
```

Создавать его самому почти не приходится: `run_kernel`, `compile_kernel`, `graph_launch` и
`graph_launch_multi` собирают его за вас, уже привязанным к буферам запуска. Поля, которые вы читаете:

| Поле / метод | Смысл |
|---|---|
| `ker.caps` | `ArchCaps` целевой платформы: `arch` и `wave_size` (64 на CDNA, 32 на RDNA, CUDA и Metal) |
| `ker.grid_x()` / `grid_y()` / `grid_z()` | `blockIdx.{x,y,z}` в виде UOp-ов `Special` (рендерятся, только если используются) |
| `ker.thread_idx` | `threadIdx.x` |
| `ker.warpid()` / `ker.laneid()` | `threadIdx / wave_size` и `threadIdx % wave_size` |
| `ker.frag(role)` | физический `RTBaseShape` для `FragRole` на этой архитектуре — паникует, если у архитектуры нет разметок матричного блока |

`block` должен содержать целое число волн (`debug_assert` в `Kernel::new`); блок запуска
обычно равен `warps * caps.wave_size`.

### Привязка глобальных буферов

```rust
// tk/src/scaffold.rs
pub fn bind_abi(&self, outputs: &[GlSpec], inputs: &[GlSpec]) -> (Vec<GL>, Vec<GL>)
pub fn gl(&self, shape: &[usize], dtype: DType) -> GL                // tk/src/tile.rs
```

`bind_abi` — это `gl` в порядке срезов: сначала выходы, потом входы, в том же порядке буферов,
что получил лаунчер. Главенствует dtype привязанного буфера; отладочная сборка проверяет, что
объявленный dtype имеет ту же ширину в байтах. Необязательный буфер (`key_lens` у FA) привязывается
замыкающим `gl` после `bind_abi`, но никогда не между.

```rust
let (outs, ins) = ker.bind_abi(
    &[GlSpec::new(&[1, 1, m, n], DType::BFloat16)],
    &[GlSpec::new(&[1, 1, m, k], DType::BFloat16), GlSpec::new(&[1, 1, n, k], DType::BFloat16)],
);
```

### Выделение тайлов

Дескрипторы формы (`tk/src/tiles.rs`) — чистые данные; обёртки (`tk/src/tile.rs`) привязывают
буфер. Сырые конструкторы принимают базовую форму явно; сокращения из scaffold разрешают её по
роли через `ker.caps`, и именно ими пользуются ядра в дереве.

| Сырой вызов | Сокращение | Что выделяет |
|---|---|---|
| `ker.rt(dims, dtype, layout, base)` | `ker.acc(dims, layout)` | f32 `RT` во фрагменте `Accumulator` |
| | `ker.acc_t(dims, layout)` | f32 `RT` в `AccumulatorT` (N-major сохранение транспонированного аккумулятора) |
| | `ker.operand(dims, dt, layout)` | 16-битный `RT` во фрагменте операнда A |
| | `ker.operand_b(dims, dt, layout)` | фрагмент операнда B (отличается от `operand` только на Metal) |
| `ker.rv(length, dtype, VecLayout::Ortho, base)` | `ker.acc_vec(length)` | f32 `RV`, `length / frag_rows` тайлов × `LaneMap::slots()` |
| `ker.st(dims, dtype, layout, base)` | `ker.shared(dims, dt, layout)` | LDS `ST` в простой полосе архитектуры |
| | `ker.shared_sw(dims, dt, layout)` | LDS `ST` в полосе с XOR-свизлом |
| `ker.st_db(..)` / `ker.st_stages(.., stages)` | `ker.shared_db(..)` / `ker.shared_sw_stages(.., stages)` | тот же тайл поверх буфера в `stages` раз больше, для программного конвейера |

`dims` — это `(rows, cols)` в элементах; по обеим осям значение должно быть кратно базовому
фрагменту (проверяется `assert`). `TileLayout::{Row, Col}` говорит, вдоль какой оси идут регистры
лейна; его читают редукции и переходы через глобальную память.

Логическая форма `RT` — `[height, width, ept]` (сетка фрагментов, затем элементы на лейн); у `ST`
она `[height, width, frag_rows, frag_cols]`. `ST::subtile(dims, (row_blk, col_blk))` — это
представление без копирования на полосу разделяемого тайла, принадлежащую одной волне;
`ST::with_base_offset(off)` выбирает стадию конвейера (`parity * st.half_elems()`).

### Упорядочивание

Тайлы — неизменяемые дескрипторы. Каждая операция возвращает целевой тайл, **заново обёрнутый**
ребром `After` на выданное ею сохранение, так что следующее чтение упорядочивается после него. Для
того, что поток данных не выражает, есть два ребра, протягиваемых руками:

- `t.after(deps)` — упорядочить следующее чтение `t` после `deps` (тайл, диапазон, барьер, массив
  или кортеж из них; `AfterDeps` в `tk/src/tile.rs`).
- `st.after(deps)` — аналог для `ST`.

---

## `Group` — вычислительный словарь

```rust
ker.warp()             // 1 wave
ker.group(n)           // 1×n waves, for collaborative GLOBAL→LDS fills
ker.group_2d(r, c)     // an r×c wave grid; group_threads = r·c·wave_size
```

`g.warp_row()` / `g.warp_col()` — координаты волны в сетке; `g.warpid_in_group()` — её плоский
индекс. Регистровые операции работают на уровне лейна и безопасны для волн в любой группе;
операции одной волны (`map_position`, редукции через `col_reduce`, перетасовки) проверяют
`warps == 1` — вызывайте их на `ker.warp()` даже в многоволновом ядре.

### Перемещение

```rust
// tk/src/group/movement.rs
pub fn load<Dst, Src: LoadInto<'k, Dst>>(&self, dst: Dst, src: Src, ix: MoveIdx) -> Src::Output
pub fn store<Dst, Src: StoreInto<'k, Dst>>(&self, dst: Dst, src: Src, ix: MoveIdx) -> Src::Output
```

Допустимые пары адресных пространств — это реализации трейтов, поэтому недопустимая пара
(`RT ← RT`) — ошибка компиляции:

| Вызов | Пара | Что выдаёт |
|---|---|---|
| `g.load(st, gl, ix)` | `ST ← GL` | коалесцированное заполнение всеми потоками группы + барьер рабочей группы |
| `g.load(rt, st, ix)` | `RT ← ST` | gather фрагмента по лейнам через `LaneMap` (на CUDA — один `ldmatrix.x4` на 16-битный фрагмент) |
| `g.load(rt, gl, ix)` | `RT ← GL` | прямой gather из глобальной памяти, без остановки в LDS |
| `g.store(st, rt, ix)` | `ST ← RT` | scatter фрагмента в LDS |
| `g.store(gl, rt, ix)` | `GL ← RT` | scatter фрагмента в глобальную память |

`MoveIdx` называет индексы по роли: `MoveIdx::block(idxs, axis)` — координата тайла в глобальном
буфере (по элементу на каждую размерность; `axis` — размерность, шаг которой покрывает строка
тайла), `MoveIdx::frag(idxs)` — смещение фрагмента на стороне регистров, `MoveIdx::at(block, frag, axis)` —
и то и другое, `MoveIdx::default()` — ничего (подтайл уже несёт свою полосу). `.masked()` ограничивает
переход `GLOBAL ↔ REG` границами тензора: на рваном краю читается `0.0`, а запись отбрасывается.

Примитивы конвейера отделяют заполнение от его синхронизации:

| Примитив | Архитектура | Применение |
|---|---|---|
| `fill_local_nobar` / `fill_local_vec_nobar` | все | заполнение без замыкающего барьера; забор ставит вызывающий |
| `stage_global_to_reg(st, gl, idxs, axis)` → `commit_regs_to_local(&[(st, stage), ..])` | все (путь AMD) | глобальные загрузки в регистры сейчас, `ds_write` в LDS потом, так что загрузки идут в полёте под MMA текущего блока |
| `cp_async_fill(st, gl, idxs, axis)` (под условием `cp_async_fill_applies`) | CUDA sm_80+ | 16-байтовый `cp.async` прямо в LDS; завершается через `cp_async_wait(n, ..)` + `.barrier(..)` |
| `war_fence2(a, b, extra)` | все | межволновой барьер, который потребляют оба gather-а, с коммитами предзагрузки в качестве зависимостей |
| `store_local_fenced(st, rt, ix, deps)` | все | scatter `RT → ST` с последующим барьером (переразметка softmax на RDNA3) |
| `store_global_with(gl, rt, ix, f)` | все | глобальное сохранение со значением `f(v, offset)` — слитые эпилоги |

### Матричное умножение

```rust
// tk/src/group/mma.rs — C += A·B over every output fragment, reducing along K
pub fn mma_ab  (&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>   // a[h, k] · b[k, w]
pub fn mma_abt (&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>   // a[h, k] · b[w, k]ᵀ
pub fn mma_atb (&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>   // a[k, h]ᵀ · b[k, w]
pub fn mma_atbt(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>
```

`a`/`b` — bf16 или f16 во фрагментах операндов, `c` — f32 во фрагменте аккумулятора (иначе
паника). Один `Op::Wmma` на шаг 16×16×16 на AMD, два `m16n8k16` на CUDA, одна операция
`simdgroup_matrix` 8×8×8 на фрагмент Apple; дескриптор берётся из таблицы `TensorCore`
планировщика, так что у ядер, написанных руками, и у действия `TC` у BEAM один источник.

### Редукции и перетасовки

```rust
// tk/src/group/reduce.rs
pub fn row_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init: f64) -> RV<'k>
pub fn col_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init: f64) -> RV<'k>
pub fn arg_reduce(&self, val: RV<'k>, idx: RV<'k>, src: &RT<'k>, dir: ArgDir) -> (RV<'k>, RV<'k>)
```

Редукция сворачивает элементы внутри лейна, а затем завершается между лейнами по `ReduceTree`
фрагмента — соседский gather через `ds_bpermute` на AMD, «бабочка» `shfl.bfly` на CUDA и Metal.
`op` — любой ассоциативный комбинатор (`|a, b| a.max(b)`, `|a, b| a.add(b)`); результат
сворачивается в `vec`, поэтому `vec` уже должен держать текущее значение.

Скалярные примитивы волны (`tk/src/group/shuffle.rs`): `wave_reduce_scalar(value, op)`,
`subgroup_reduce_scalar(value, width, op)`, `broadcast_scalar(value, lane)` и тайловые формы
`shuffle_xor`, `shuffle_down`, `shuffle_up`, `compare_exchange` (стадии битонной сортировки).
Ни один из них не трогает LDS.

### Поэлементные операции

| Вызов | Смысл |
|---|---|
| `g.zero(rt)` / `g.ones(rt)` / `g.neg_inf(rt)`; `zero_rv` / `clear_rv(rv, v)` | заполнение константой |
| `g.copy(dst, &src)` | поэлементное копирование с приведением при несовпадении dtype |
| `g.transpose(dst, &src)` | поменять местами `height` и `width` сетки фрагментов |
| `g.map(t, \|x, idx\| ..)` | применить UOp-выражение к каждому элементу |
| `g.map_position(rt, row_blk, col_blk, \|x, idx, row, col\| ..)` | то же с глобальными `(row, col)` элемента, прочитанными из `LaneMap` |
| `g.mask_where(rt, row_blk, col_blk, fill, \|row, col\| pred)` | `where(pred, fill, x)` — каузальная маска и маска паддинга |
| `g.add/sub/mul/div/maximum(a, &b)`, `*_scalar(a, s)`, `*_rv(rt, &rv)`, `g.exp2(t)` | математика над тайлами (`tk/src/math.rs`) |

Операторный сахар (`tk/src/ops.rs`) ведёт к тем же вызовам, так что тело читается как математика:
обновление онлайн-softmax в `tk/src/kernels/fa.rs` выглядит так:

```rust
let scale_vec = (max_vec_last - &max_vec).exp2();
o_reg = o_reg * &scale_vec;
norm_vec = norm_vec * &scale_vec;
let att = (att - &max_vec).exp2();
```

`T op &T` — операнды одной формы, `RT op &RV` транслирует вектор вдоль оси разметки тайла,
`T op f64` — скаляр.

---

## Циклы

```rust
// tk/src/loop_scope.rs
let lp = ker.loop_static(trips);          // a tracked RANGE with a constant trip count
let lp = ker.loop_dynamic(bound_uop);     // a runtime trip count (FA's causal block-skip)
lp.index()                                 // the counter, for addressing
lp.reinit(t)                               // t.after(range): re-run a per-trip init inside the loop
lp.close()                                 // end the last store around the range; returns the END
lp.close_carry(t)                          // close and rebind one carried tile to its post-loop value
lp.close_barrier(commits)                  // close with a workgroup fence folded into the END
```

Два правила, которые область цикла делает незабываемыми:

- Переинициализация на каждой итерации (`g.zero(acc)` в начале тела) должна зависеть от счётчика
  цикла, иначе линеаризатор поднимет её над циклом, и аккумулятор потащит устаревшее состояние.
  Пишите `g.zero(lp.reinit(acc))`.
- `RANGE` допускает ровно один `END`. Если в одном цикле несколько аккумуляторов, сцепите остальные
  в одно замыкающее сохранение (GEMM протягивает вход A каждого аккумулятора через MMA предыдущего),
  а затем читайте каждое итоговое значение как `acc.after(&ended)`.

`Kernel::range` / `range_uop` / `endrange` / `endrange_to` / `endrange_barrier_to` — сырые формы,
которые оборачивает `Loop`; граф они выдают идентичный.

---

## Завершение и запуск

```rust
pub fn finish(&self, stores: usize) -> Arc<UOp>          // tk/src/kernel.rs
```

`finish(n)` снимает последние `n` терминальных сохранений — по одному на выходной глобальный
буфер, — закрывает вокруг каждого ещё открытый отслеживаемый диапазон и замыкает их в `SINK` с
`KernelInfo { opts_to_apply: Some(vec![]), name: Some(name) }`. Ядро, которое оставляет диапазон
открытым к моменту `finish`, должно иметь ровно одно сохранение. Сохранения попадают в стек через
операции перемещения или явно через `ker.push_store(store, buf)` (так группирует свои векторные
сохранения линейное ядро нормализации).

```rust
// tk/src/launch.rs
pub fn graph_launch(name, grid, block, out: Tensor, ins: &[&Tensor], caps: ArchCaps, build) -> Result<Tensor>
pub fn graph_launch_multi(name, grid, block, outs: Vec<Tensor>, ins, caps, build) -> Result<Vec<Tensor>>
pub fn launch_custom<T>(device, archs: ArchSet, validate, applies, build) -> Result<Option<T>>
pub fn run_kernel(name, grid, block, outs: &mut [&mut Tensor], ins: &[&Tensor], build) -> Result<()>
pub fn compile_kernel(name, grid, block, outs, ins, build) -> Result<CompiledLaunch>
```

`graph_launch` оборачивает `SINK` в узел `Op::Call` и возвращает ленивый тензор; `out` — это
`Tensor::empty(shape, dtype)`, а заглушки, которые видит тело, — `[out, ins...]`, в порядке
`bind_abi`. `launch_custom` — трёхвариантная политика, которой следует каждое библиотечное ядро
([Пишем прямо в IR](./lowering)): `resolve_supported_arch` по `ArchSet` ядра (`Ok(None)` вне его),
`validate(arch)` (`Err` для некорректного запроса), `applies(arch)` (`Ok(None)`, если форма не
бьётся на тайлы), затем `build(arch)`.

```rust
// tk/src/kernels/norm.rs — the shape of every graph-native entry
crate::launch_custom(
    &x.device(),
    NORM_SUPPORTED_ARCHS,
    move |_arch| check_norm_operands("rms-norm", &check.0, &check.1, &check.2, check.3),
    move |arch| select_norm_cfg(rows, d, crate::ArchCaps::for_arch(arch).wave_size).is_some(),
    move |arch| {
        let caps = crate::ArchCaps::for_arch(arch);
        let cfg = select_norm_cfg(rows, d, caps.wave_size).expect("checked by the fit predicate");
        let (grid, block) = launch_dims(rows, cfg.rows_per_block, caps.wave_size);
        let out = Tensor::empty(&xd, dtype.clone());
        let dt = dtype.clone();
        crate::graph_launch("rms_norm", grid, block, out, &[x, weight], caps, move |ker| {
            build_row_norm(ker, rows, d, dt, eps, cfg, false);
            ker.finish(1)
        })
    },
)
```

`run_kernel` / `compile_kernel` — лицо DEBUG с прямым диспатчем; см. [Отладка](./debugging).

---

## Ниже тайлов

Некоторым ядрам нужны адреса, а не тайлы. `tk/src/index.rs` — слой плоской адресации, на котором
построена каждая операция над тайлами, и он публичный: `Idx` (`Const(i64)` или `Uop`), `flat_index(buf,
shape, idxs)`, `load_at`, `load_off`, `load_off_vec(buf, off, w)` / `store_off_vec` (одно обращение
шириной `w` на лейн, которое рендерер сворачивает в 128-битную инструкцию) и формы с условием
`load_off_gated` / `index_off_gated`. Ядро RMS-norm и пролог QKV-norm-RoPE для Qwen3 в
`model/src/qwen3/tk/mod.rs` написаны целиком на этом уровне — одна волна на строку, ни `RANGE`, ни
LDS — и переиспользуют строковый словарь нормализации (`plan`, `vload`, `vpick`, `vstore`,
`inv_rms`, `scale_by`).

`tk/src/asm.rs` открывает рычаги машинного планировщика AMD как типизированные узлы `Op::Custom`,
привязанные к зависимости: `s_setprio(prio, dep)`, `s_waitcnt_lgkmcnt(n, dep)`,
`sched_barrier(mask, dep)`, `iglp_opt(mode, dep)`. GEMM использует `sched_barrier(0, ..)` на
gfx12, где `ArchCaps::needs_pipeline_commit_fence()` сообщает, что иначе планировщик бэкенда
поднял бы LDS-коммит конвейера над MMA итерации.

`tk/src/grid.rs::l2_swizzle(wgid, num_wgs, grid_m, grid_n)` отображает плоский id рабочей группы
в `(pid_m, pid_n)`, чтобы одновременно запланированные рабочие группы делили L2 одного XCD
(чиплетное преобразование HipKittens); включается через `GemmCfg::l2_swizzle`.
