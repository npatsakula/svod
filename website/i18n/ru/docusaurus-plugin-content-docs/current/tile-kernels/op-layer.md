---
sidebar_label: Слой операций
---

# Слой операций {#op-layer}

`svod_tk3::ops` — то, что вызывают модели. Каждая операция возвращает ленивый `Tensor`. Когда
устройство, типы данных и формы подходят ядру, результат вычисляет ядро tk3. Иначе операция
сама строит эквивалентный граф. Модель никогда не проверяет, применимо ли ядро, и никогда не
дополняет данные до размера тайла.

```rust
pub fn linear(x: &Tensor, w: &Tensor, opts: Linear) -> Result<Tensor>;
pub fn conv2d(x: &Tensor, w: &Tensor, opts: Conv) -> Result<Tensor>;
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, opts: Attn) -> Result<Tensor>;
pub fn heads(qkv: &Tensor, opts: Qkv) -> Result<(Tensor, Tensor, Tensor)>;
pub fn layer_norm(x: &Tensor, w: &Tensor, b: Option<&Tensor>, eps: f64) -> Result<Tensor>;
pub fn add_layer_norm(x: &Tensor, residual: &Tensor, w: &Tensor, b: Option<&Tensor>, eps: f64)
    -> Result<(Tensor, Tensor)>;
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor>;
pub fn add_rms_norm(x: &Tensor, residual: &Tensor, w: &Tensor, eps: f64) -> Result<(Tensor, Tensor)>;
pub fn supported(device: &DeviceSpec) -> bool;
```

| Операция | Формы | Опции |
|---|---|---|
| `attention` | `q [B, T, H, D]`, `k`/`v [B, Tk, H_kv, D]` → `[B, T, H, D]` | `Attn { causal, keys, window, seg_start, cache, splits, scale, bias }` |
| `linear` | `x [lead..., K]`, `w [N, K]` → `[lead..., N]` | `Linear { bias, act, gated, residual, scale }`; gated `w` имеет форму `[2N, K]` |
| `conv2d` | `x [B, H, W, Cin]`, `w [Cout, kh, kw, Cin / groups]` → `[B, Ho, Wo, Cout]` | `Conv { stride, pad, dilation, groups, bias, act, residual, scale, out_dtype }`; свёртка 1×1 — это `linear` |
| `heads` | `qkv [B, T, (H + 2·H_kv)·D]` → `q`, `k`, `v` | `Qkv { heads, kv_heads, head_dim, q_norm, k_norm, eps, rope }` |
| `layer_norm`, `rms_norm` | `x [..., D]`, `w`/`b [D]` | `add_*` принимают `residual` той же формы, что `x`, и возвращают `(x + residual, norm)` |

`Linear::scale` умножает значение после активации до прибавления residual: `scale·act(x·wᵀ + bias) + residual`, так что полушаг Conformer `x + 0.5·ffn(x)` — это один GEMM.

`Attn::keys` — это `KeyMask::None`, `KeyMask::Lens(&lens)` (`[B]` — число валидных ключей) или
`KeyMask::Bool(&mask)` (`[B, Tk]`, true там, где ключ учитывается). `Attn::cache` принимает
`Cache { head_start, kv_heads, row_map, appended }`, причём `appended` требует `KeyMask::Lens`.
`Attn::scale` по умолчанию равен `1/√D`. `Attn::bias` (`[B, H, T, Tk]` или `[1, H, T, Tk]` в типе потока) прибавляется к масштабированным оценкам до масок, как относительное позиционное смещение WavLM. `Qkv::rope` — это `(cos, sin)` формы `[1, T, 1, D/2]`
(по позиции) или `[B, T, 1, D/2]` (по токену). Маски описаны вместе с
[ядром внимания](./kernel-library#flash-attention).

## Ядро или граф {#kernel-or-graph}

Решение — чистая функция в `ops::shape`, `fn(target, dtypes, extents, …) -> Plan<Cfg>`, поэтому
оно тестируется на хосте без GPU. `Plan::Kernel(candidates)` перечисляет конфигурации, первой —
выбор без тюнинга. `Plan::Graph(Fallback)` говорит, почему работает граф.

| `Fallback` | Когда |
|---|---|
| `Target` | Тензор не на устройстве по умолчанию, или у устройства нет таблиц tk3 (сегодня — всё, кроме CUDA sm_80+) |
| `Dtype` | Какой-либо операнд в f32, или операнды не имеют общего 16-битного типа с матричным ядром |
| `Symbolic` | Символьна размерность, отличная от связанной ведущей (для `linear` — также символьное `N`) |
| `Shape` | `linear`: `N` не кратно 8 или нет строк. `attention`: `D ∉ {48, 64, 128}` или пустая размерность. `heads`: `D` не степень двойки в 16..=256. Нормализации: `D` не степень двойки в 256..=2048. `conv2d`: `groups > 1`, `Cin` не кратно 16, `Cout` не кратно 8 или пустой выход |
| `Config` | Ни одна тайловая конфигурация не подходит. Для `linear` — `K` не кратно 16 |

Реальные случаи из `test/unit/ops_plan.rs` на цели sm_86:

```rust
#[test_case(&[4096, 4096], 4096, false, Ok(BIG); "large grid, deepest ring")]
#[test_case(&[8, 37, 512], 512, false, Ok(SMALL); "medium grid")]
#[test_case(&[37, 40], 96, false, Err(Fallback::Config); "k off every bk")]
#[test_case(&[37, 64], 100, false, Err(Fallback::Shape); "n not a multiple of 8")]
fn linear_plans(x: &[usize], n: usize, gated: bool, want: Result<GemmCfg, Fallback>) {
    assert_eq!(first(shape::linear(Some(&sm86()), &[BF16, BF16], Some(&ext(x)), n, gated)), want);
}
```

:::note[Почему f32 остаётся на графе]
Ядра принимают только 16-битные операнды. Приведение модели в f32 вниз обменяло бы около трёх
десятичных знаков точности на скорость, поэтому слой операций оставляет этот выбор за типом
данных модели.
:::

## Переменные батча {#batch-variables}

Размерность 0 операнда может быть символьной, если она связана с переменной времени
выполнения. Такая переменная — либо `batch_var` из JIT, либо `DefineVar`/ограниченный `Param` с
минимумом и максимумом. Тогда ядро запускается на живое количество по оси z сетки, а его буферы
рассчитаны на максимум. Выходы выделяются на полную ёмкость с сохранением символьной размерности
в форме (`Tensor::empty_dynamic`). Поэтому следующее ядро привязывает сам реализованный буфер, и
копия для его сжатия не делается. Любая другая символьная размерность — это
`Fallback::Symbolic`.

## Ошибки {#errors}

`Err` зарезервирован для того, что отверг бы и графовый вариант операции, плюс неудачная сборка
ядра:

| `ops::Error` | Значение |
|---|---|
| `Shape { op, operand, got, expected }` | Форма операнда не подходит операции |
| `Dtype { op, operand, got, want }` | `w` (или `k`, `v`, веса нормализации, таблицы rope) по типу отличается от входа |
| `Heads { op, heads, kv_heads }` | `heads` не кратно `kv_heads` |
| `Graph { op, source }` | Не удалось построить графовый запасной вариант |
| `Launch { op, source }` | Не удалось понизить или привязать ядро |

## В модели {#in-a-model}

Self-attention из Nemotron-3-Diarization, `model/src/nemotron_diar/model.rs`: одна общая
проекция QKV, слитый пролог с RoPE, внимание с длинами ключей и выходная проекция с residual в
эпилоге.

```rust
fn forward(&self, x: &Tensor, rope: &(Tensor, Tensor), key_lens: &Tensor, residual: &Tensor) -> Result<Tensor> {
    let (b, s, d) = (x.dim(0)?, x.dim(1)?, x.dim_const(2)?);
    let qkv = ops::linear(x, &self.qkv_weight, ops::Linear::default())?;
    let (cos, sin) = rope;
    let split = Qkv {
        heads: self.num_heads,
        kv_heads: self.num_heads,
        head_dim: d / self.num_heads,
        q_norm: None,
        k_norm: None,
        eps: 0.0,
        rope: Some((cos, sin)),
    };
    let (q, k, v) = ops::heads(&qkv, split)?;
    let opts = Attn { keys: KeyMask::Lens(key_lens), ..Attn::default() };
    let out = ops::attention(&q, &k, &v, opts)?;
    project(&self.o_proj, &out.try_reshape([b, s, SInt::Const(d)])?, Act::None, Some(residual))
}
```

`project` — это `ops::linear` со смещением слоя, функцией активации и необязательным residual.
Тот же файл пропускает свои LayerNorm через `ops::layer_norm`.
