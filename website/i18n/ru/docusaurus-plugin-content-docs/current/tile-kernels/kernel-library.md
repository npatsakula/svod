---
sidebar_label: Библиотека ядер
---

# Библиотека ядер

Лицо USE: каждое ядро, которое поставляет `svod-tk`, вызывается с обычными тензорами и без всякого
знания о тайлах. Каждое возвращает ленивый `Tensor` (узел `Op::Call`), который встраивается в граф
модели и реализуется обычным путём `prepare()`, и каждое следует трёхвариантному контракту из главы
[Пишем прямо в IR](./lowering):

| Результат | Значение |
|---|---|
| `Ok(Some(out))` | ядро отработало |
| `Ok(None)` | ядро неприменимо: устройство вне `ArchSet` ядра, нет его LLVM-бэкенда или форма не тайлится — откатывайтесь осознанно |
| `Err(LaunchError)` | запрос некорректен (dtype, ранг, символьная размерность, правило делимости) — ошибка вызывающего кода |

Операнды — bf16 или f16, если не оговорено иное; накопление в f32.

---

## Целевые архитектуры {#targets}

Каждое ядро объявляет свой `ArchSet` (`tk/src/target.rs`): явный список AMD плюс открытый сверху
нижний порог CUDA capability и нижний порог семейства Apple GPU.

| Ядро | gfx942 (CDNA3) | gfx1151 (RDNA3.5) | gfx1200 / gfx1201 (RDNA4) | CUDA sm_80+ | Metal Apple7+ |
|---|---|---|---|---|---|
| `flash_attention` / `_with` / `_tuned` | да | да | да | да | да |
| `matmul` (квадратный) | да | да | да | да | да |
| `gemm_nt` / `_with` / `_with_epilogue` | — | да | да | да | — |
| `rms_norm` / `add_rms_norm` | — | да | да | да | — |
| `single_query_attention` / `_packed` | да | да | да | да | — |
| `knn` | да | да | да | — | — |
| `kmeans_assign` | да | да | да | — | — |

Константы — `FA_SUPPORTED_ARCHS`, `MATMUL_SUPPORTED_ARCHS`, `GEMM_NT_SUPPORTED_ARCHS`,
`NORM_SUPPORTED_ARCHS`, `SQ_ATTENTION_SUPPORTED_ARCHS`, `KNN_SUPPORTED_ARCHS` и
`KMEANS_SUPPORTED_ARCHS` в `tk/src/kernels/`. Семейство присоединяется к ядру через валидацию и
замер собственной таблицы тайлов: `gemm_nt` и нормализации работают только на wave32 потому, что
таблицу для wave64 под них никто не замерял, а не потому, что тело ядра там не может исполняться.
`flash_attention_supported(&device)` отвечает только на вопрос об архитектуре — для вызывающего
кода, который дополняет или группирует длину последовательности до запуска.

---

## Flash attention

```rust
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> LaunchResult<Option<Tensor>>
pub fn flash_attention_with(q, k, v, opts: FaOpts) -> LaunchResult<Option<Tensor>>
pub fn flash_attention_tuned(q, k, v, opts, policy: impl Fn(&DeviceSpec, GpuArch) -> FaPolicy + Copy) -> ..

pub struct FaOpts<'a> {
    pub causal: bool,                      // default true
    pub key_lens: Option<&'a Tensor>,      // [B] i32 valid-key counts: keys >= key_lens[b] are masked
    pub seg_start: Option<&'a Tensor>,     // [B, N] i32: query q of batch b sees no key before seg_start[b, q]
}
```

`q` имеет форму `[B, N, H, D]`, `k`/`v` — `[B, N, H_kv, D]` (GQA: `H % H_kv == 0`), выход —
`[B, N, H, D]` в dtype операндов. Раскладка «сначала последовательность», а не «сначала головы»:
модель решейпит проекцию прямо в неё, без транспонирования.

- `Ok(None)`: архитектура вне набора; `N` не кратно `q_blk · 8` (Q-тайл одного варпа, умноженный
  на восемь волн рабочей группы; `FLASH_ATTENTION_SEQUENCE_MULTIPLE` — базовое значение
  `128`); длина KV отличается от `N` (cross-attention не реализован); размер головы, при котором
  двойные буферы K/V-тайлов не помещаются в разделяемую память устройства.
- `Err`: dtype вне `{bf16, f16}` или разный у `q` и `k`/`v`; `D % 16 != 0`;
  `H % H_kv != 0`; форма `k`/`v`, отличная от `[B, N, H_kv, D]`.

`key_lens` маскирует только ключи: дополненные строки запросов всё равно вычисляются, и вызывающий
код их отбрасывает. `key_lens[b] == 0` приводится к `1`, чтобы строка осталась конечной. `seg_start`
упаковывает несколько последовательностей в одну строку: каждое значение должно лежать в `0..=q` и
оставлять хотя бы один видимый ключ. Разобранный пример — глава [Flash Attention](./flash-attention);
тайл варпа замеряется при первом использовании ([Автотюнинг](./tuning)).

---

## GEMM

```rust
pub fn matmul(a: &Tensor, b: &Tensor) -> LaunchResult<Option<Tensor>>              // [n, n] · [n, n] → f32
pub fn gemm_nt(x: &Tensor, w: &Tensor) -> LaunchResult<Option<Tensor>>             // [lead..., K] · [N, K]ᵀ → [lead..., N]
pub fn gemm_nt_with(x, w, cfg: impl Fn(usize, usize, usize) -> Option<GemmCfg> + Copy) -> ..
pub fn gemm_nt_with_epilogue(x, w, epilogue: Epilogue<&Tensor>) -> LaunchResult<Option<Tensor>>

pub enum Epilogue<T> {
    Plain,               // y = x·wᵀ
    Add(T),              // y = x·wᵀ + residual, residual [lead..., N] in the operand dtype
    SwiGlu { pair: usize }, // y = silu(gate)·up off a fused [2I, K] gate/up weight; y is [lead..., N/2]
}
pub fn swiglu_pair_width(spec: &DeviceSpec) -> Option<usize>
```

`matmul` — эталонное квадратное ядро: на вход любой float dtype (приводится к bf16), на выход f32,
любая архитектура. Это индикатор производительности DSL, а не продакшн-GEMM.

`gemm_nt` — продакшн-линейный слой. `x` имеет форму `[lead..., K]` любого ранга ≥ 2 (активация
`[B, L, K]` привязывается без решейпа и копирования), `w` — `[N, K]`, как вес и хранится, а `y` —
`[lead..., N]` в dtype операндов: f32-аккумуляторы сужаются в регистрах, так что промежуточного
f32 в памяти нет. `M = ∏lead` и `N` должны быть кратны 64, а `K` — кратна 32-элементной полосе,
причём полос не меньше двух; иначе `Ok(None)`, и вызывающий код дополняет до 128 или использует
`Tensor::linear`.

Ради эпилогов ядро и существует: они сворачивают проход, который граф заплатил бы после GEMM, в его
запись. `Add` читает остаток по тому же смещению, что и запись, и складывает в выходном dtype — ровно
с тем округлением, что и `try_add` в графе. `SwiGlu` требует, чтобы строки слитого веса шли
чередующимися блоками gate/up по `pair` строк; `swiglu_pair_width(&device)` равно `reg_n / 2` из
таблицы тайлов устройства или `None`, если тайлы расходятся (тогда вызывающий код оставляет
отдельный проход SwiGLU). Модель загружает вес в этом порядке один раз, потому что `M` выбирает
тайл в момент запуска, и каждый тайл-кандидат должен читать одну и ту же раскладку.

Тайл берётся из `GemmPolicy` (`tk/src/kernels/gemm.rs`): таблица на семейство —
`CUDA_TILES`, `RDNA_TILES`, `RDNA4_TILES`, — замеряемая при первом использовании для каждой формы и
эпилога ([Автотюнинг](./tuning)), или статический выбор `GemmPolicy::cfg`, когда тюнинг выключен.
`gemm_nt_with` принимает функцию выбора от вызывающего кода (так её перебирают бенчмарки).

---

## RMS norm

```rust
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> LaunchResult<Option<Tensor>>
pub fn add_rms_norm(x, residual: &Tensor, weight, eps) -> LaunchResult<Option<(Tensor, Tensor)>>   // (h, y)
pub fn select_norm_cfg(rows: usize, d: usize, lanes: usize) -> Option<NormCfg>
```

`x` имеет форму `[rows..., D]`, `weight` — `[D]`, оба в одном 16-битном dtype. Одна волна на
строку, строка держится в регистрах, сумма квадратов досчитывается butterfly-шаффлом — без LDS,
без барьера, без `RANGE`. Численно ядро повторяет граф операция в операцию:
`y = dtype((f32(x) · rsqrt(Σx²/D + eps)) · f32(w))`, одно округление в конце; отличается только
порядок суммирования. `Ok(None)`, если `D` не кратно ширине волны или превышает 64 элемента на
лейн (`2048` при wave32).

`add_rms_norm` возвращает `(h, y)`, где `h = x + residual` округлено так же, как округляет сложение
в графе, а `y = rms_norm(h)`, так что pre-norm слой декодера пишет и читает свой остаточный поток
один раз. Если предыдущая проекция уже приняла остаток через `Epilogue::Add`, достаточно двухпроходного
`rms_norm`; `model/src/qwen3/decoder_layer.rs` выбирает между ними для каждого слоя.

---

## Single-query attention

```rust
pub fn single_query_attention(q, k, v, opts: SqAttentionOpts<'_>) -> LaunchResult<Option<Tensor>>
pub fn single_query_attention_packed(q, k, v, head_offset: usize, opts) -> LaunchResult<Option<Tensor>>

pub struct SqAttentionOpts<'a> {
    pub key_lens: Option<&'a Tensor>,                 // [B] i32, entries in 0..=N
    pub include_last: bool,                           // also score key N-1 (Whisper's self-cache slot)
    pub appended: Option<(&'a Tensor, &'a Tensor)>,   // the step's own [B, 1, H, D] K/V, scored after the prefix
    pub split: Option<usize>,                         // K/V chunks; None = the device's SqPolicy, tuned on first use
    pub cache_map: Option<&'a Tensor>,                // [B] i32: which K/V row each query row reads
}
```

Ядро шага декодирования: `q` имеет форму `[B, 1, H, D]` в f32, `k`/`v` — `[B, N, H_total, D]` в
f32, f16 или bf16 (или `[1, N, H_total, D]`, чтобы обслуживать все строки из одного кэша), выход —
`[B, 1, H, D]` в f32.
Одна волна владеет одной парой `(batch, head)`; `Q` остаётся в регистрах, пока K/V текут по `N`;
скалярные произведения — all-reduce через XOR-шаффлы, а softmax — однопроходное онлайн-обновление.
Нет ни LDS, ни матричного ядра, поэтому его `ArchSet` — самый широкий список AMD плюс CUDA.

`_packed` выбирает головы `head_offset..head_offset + H` из упакованного кэша, не нарезая его.
Длинное attention без маски делит K/V на непрерывные куски, по волне на кусок, и сливает их
состояния softmax во втором проходе; `SqPolicy` подбирает разбиение по бюджету резидентных волн
устройства и при первом использовании замеряет ближайшие делители.

---

## k-NN и k-means

```rust
pub fn knn(x: &Tensor, c: &Tensor, k: usize) -> LaunchResult<Option<(Tensor, Tensor)>>           // (dists [N, k] f32, idxs [N, k] i32)
pub fn kmeans_assign(x: &Tensor, c: &Tensor) -> LaunchResult<Option<(Tensor, Tensor)>>         // (cluster_ids [N] i32, best_dist [N] f32)
pub fn kmeans_update(x, cluster_ids, old_centroids) -> LaunchResult<(Tensor, Tensor)>           // (new_centroids [K, D], shift [K])
```

Оба ядра прогоняют корпус (центроиды) через матричное ядро и ведут текущий top-K (argmin) по
оценке без x², `‖c‖² − 2⟨x, c⟩`, так что матрица расстояний `[N, M]` никогда не формируется;
хостовая сторона приводит к bf16, дополняет `D` (и `N`) до границы WMMA и добавляет обратно `‖x‖²`
ради точных f32-расстояний. `knn` принимает `k` в `1..=16`. `kmeans_update` — чистая графовая
операция (`scatter_reduce` по кластерам, исправление пустых кластеров, сдвиг по кластерам), потому что
паттерн sort/scatter не тайлится; цикл Ллойда остаётся за вызывающим кодом. Только AMD.

---

## Ядро в модели

Политика — какое ядро и какой откат — принадлежит модели, а не ядру. Эталонная интеграция —
декодер Qwen3 в `model/src/qwen3/`:

```rust
// model/src/qwen3/linear.rs
pub(crate) fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    if fusable(x, w)
        && let Some(y) = svod_tk::gemm_nt(x, w).context(TkSnafu)?
    {
        return Ok(y);
    }
    Ok(x.contiguous().linear().weight(w).call()?)
}
```

```rust
// model/src/qwen3/attention.rs
if matches!(q.dtype().base(), ScalarDType::Float16 | ScalarDType::BFloat16)
    && let Some(out) =
        svod_tk::flash_attention_with(q, k, v, svod_tk::FaOpts { causal: true, key_lens: None, seg_start })
            .context(TkSnafu)?
{
    return Ok(out);
}
// else: permute to head-major and run scaled_dot_product_attention
```

Три привычки, которые стоит перенять:

- **Сначала проверяйте `fusable`.** `Err` означает некорректный запрос, и проверка на 16-битность
  на стороне вызывающего кода не даёт законному f32-пути быть принятым за ошибку.
- **Пробрасывайте ошибку.** `LaunchError` упаковывается в enum ошибок модели
  (`#[snafu(source(from(svod_tk::LaunchError, Box::new)))]`), так что неудачная сборка становится
  ошибкой модели с приложенным контекстом ядра.
- **Готовьте форму под ядро заранее.** `embed.rs` группирует дополненные длины последовательностей
  до кратных `FLASH_ATTENTION_SEQUENCE_MULTIPLE`, а `feed_forward.rs` при загрузке чередует вес
  gate/up по `swiglu_pair_width`, так что ядра применяются, а не отказываются.

Слияние, знающее раскладку памяти модели, живёт рядом с моделью: `model/src/qwen3/tk/mod.rs` —
пролог QKV-norm-RoPE, написанный на строковом словаре ядра нормализации, закрытый проверкой
`NORM_SUPPORTED_ARCHS` и запускаемый через `graph_launch_multi` с тремя выходами. Действует та же
политика `launch_custom`: это tk-ядро во всех отношениях, кроме места, где оно лежит.
