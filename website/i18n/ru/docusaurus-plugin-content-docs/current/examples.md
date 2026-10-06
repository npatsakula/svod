---
sidebar_label: Тензорный API
---

# Тензорный API на примерах

`svod-tensor` — ленивый тензорный слой, на котором написана каждая модель Svod.
Эта страница проходит по нему в том порядке, в каком вы будете им пользоваться:
построить граф, выполнить его (realize), прочитать результат, а затем скомпилировать граф
один раз и запускать его повторно. Готовые модели описаны в разделе
[Запуск моделей](./models), файлы `.onnx` — в разделе [ONNX-инференс](./onnx).

```toml
[dependencies]
svod-tensor = "0.1"
svod-dtype  = "0.1"   # DType
ndarray     = "0.17"            # array!, views
```

Крейты опубликованы на crates.io; чтобы следить за `main`, укажите
`git = "https://github.com/npatsakula/svod"` вместо версии.

**Главное правило:** операции строят граф и ничего не выполняют. `realize()`
компилирует и выполняет граф; `prepare()` компилирует его в план, который можно
выполнять сколько угодно раз.

---

## Первый тензор

```rust
use svod_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);

    let sum = (&a + &b)?;          // nothing runs yet
    let scaled = (&sum * 0.1)?;    // a scalar is a valid operand

    scaled.realize()?;             // schedule, compile, execute
    println!("{:?}", scaled.as_vec::<f32>()?);   // [1.1, 2.2, 3.3, 4.4]
    Ok(())
}
```

- `Tensor::from_slice` принимает всё, что реализует `AsRef<[T]>`, — массив, `Vec`,
  срез — и копирует данные в буфер на устройстве по умолчанию.
- Бинарные операторы возвращают `Result<Tensor>`: несовпадение формы или dtype —
  исправимая ошибка, отсюда `?`. Каждый операнд может быть `&Tensor` или
  `Tensor` во владении; правым операндом может быть и скаляр, который
  материализуется в dtype тензора. Скаляру слева нужен явный тип:
  `2.0f32 * &a`. Унарный `-&a` не может завершиться ошибкой и возвращает
  обычный `Tensor`.
- `realize(&self)` планирует весь граф, стоящий за тензором, сливает всё, что
  можно слить, компилирует ядра и запускает их. Выполненный тензор остаётся
  доступен по разделяемой ссылке.

Чтение данных:

| Метод | Выполняет граф? | Возвращает |
|---|---|---|
| `as_vec::<T>()`, `as_ndarray::<T>()` | никогда — ошибка `NoBuffer`, если граф не выполнен | копию во владении |
| `to_vec::<T>()`, `to_ndarray::<T>()` | по необходимости | копию во владении |
| `item::<T>()` | по необходимости | единственный элемент |
| `array_view::<T>()`, `array_view_mut::<T>()` | никогда | заимствованное представление `ndarray` без копирования |

Самая короткая форма — `(&a + &b)?.to_vec::<f32>()?`. Семейство `as_*` нужно
там, где скрытое выполнение графа было бы ошибкой. Представлениям нужен буфер,
отображаемый в память хоста (CPU или GPU-буфер с отображением на хост), и
тензор, который владеет своим буфером, а не является представлением чужого.

---

## Формы и broadcasting

```rust
use ndarray::array;
use svod_tensor::Tensor;

fn shapes() -> Result<(), Box<dyn std::error::Error>> {
    let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("{:?}", data.dims()?);                       // [6]

    let matrix = data.try_reshape(&[2, 3])?;              // [[1, 2, 3], [4, 5, 6]]
    let same = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(matrix.dims()?, same.dims()?);

    let transposed = matrix.try_transpose(0, 1)?;         // [3, 2]

    // [3, 2] + [1, 2] -> [3, 2]: the row vector is added to every row
    let bias = Tensor::from_ndarray(&array![[100.0f32, 200.0]]);
    let biased = (&transposed + &bias)?;
    println!("{:?}", biased.to_ndarray::<f32>()?);
    // [[101, 204],
    //  [102, 205],
    //  [103, 206]]
    Ok(())
}
```

| Операция | Действие |
|---|---|
| `try_reshape(&[2, 3])` | Новая форма с тем же числом элементов |
| `try_reshape(&[-1, 3])` | `-1` выводит размер оси из общего числа элементов |
| `try_transpose(0, 1)` | Меняет местами две оси |
| `try_permute(&[0, 2, 1])` | Переупорядочивает все оси |
| `try_squeeze(Some(dim))` / `try_squeeze(None)` | Удаляет одну ось размера 1 или все такие оси |
| `try_unsqueeze(dim)` | Вставляет ось размера 1 |
| `try_expand(&[3, 2])` | Растягивает ось размера 1 без копирования |
| `try_shrink([(0, 2), (1, 3)])` | Вырезает диапазон по каждой оси |
| `try_pad(&[(1, 1), (0, 0)])` | Дополняет нулями по каждой оси |
| `Tensor::cat(&[&a, &b], dim)` / `Tensor::stack(..)` | Конкатенация / стекирование |

Отрицательные номера осей везде отсчитываются с конца. `Tensor::from_ndarray`
копирует массив один раз; несмежный массив проходит через промежуточный `Vec`.

**Просмотр формы.** `dims()` возвращает `Vec<usize>` и завершается ошибкой, если
хотя бы одна ось символьная; `dim(axis)` возвращает `SInt` (константу или
символьное значение); `dim_const(axis)` возвращает `usize` или `NonConstDim`;
`shape()` — это вся `Shape`. `dtype()` и `device()` не могут завершиться
ошибкой. `Tensor` реализует `Debug` и выводит только метаданные —
`Tensor { shape: [4], dtype: Scalar(Float32), device: Cpu, realized: false }` —
и никогда не выводит данные, так как это потребовало бы чтения с устройства.

**Broadcasting** работает как в NumPy: формы выравниваются справа, и каждая ось
должна совпадать или иметь размер 1.

```text
[3, 2] + [1, 2] -> [3, 2]
[3, 2] + [2]    -> [3, 2]   (implicit [1, 2])
[3, 2] + [3]    -> error    ("cannot broadcast shapes", reported as ErrorKind::UOp)
```

---

## Умножение матриц

```rust
use ndarray::array;
use svod_tensor::Tensor;

fn matmul() -> Result<(), Box<dyn std::error::Error>> {
    // 4 samples with 3 features each
    let input = Tensor::from_ndarray(&array![
        [1.0f32, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]);
    // 3 features -> 2 outputs
    let weights = Tensor::from_ndarray(&array![[0.1f32, 0.2], [0.3, 0.4], [0.5, 0.6]]);

    let output = input.dot(&weights)?;                   // [4, 3] @ [3, 2] -> [4, 2]
    println!("{:?}", output.to_ndarray::<f32>()?);
    Ok(())
}
```

`dot` (псевдоним `matmul`) свёртывает последнюю ось левого операнда с
предпоследней осью правого; ведущие batch-оси подчиняются broadcasting.

| Левый | Правый | Результат |
|---|---|---|
| `[M, K]` | `[K, N]` | `[M, N]` |
| `[K]` | `[K, N]` | `[N]` |
| `[M, K]` | `[K]` | `[M]` |
| `[B, M, K]` | `[K, N]` или `[B, K, N]` | `[B, M, N]` |

Несовпадение `K` даёт `DotShapeMismatch`. `matmul_with().other(&w).dtype(DType::Float32).call()`
задаёт dtype аккумулятора, что важно для входов в fp16/bf16.

---

## Небольшой классификатор

`nn::Linear` вычисляет `x @ W.T + b` с раскладкой весов `[out, in]`, как в PyTorch.
`sequential` выстраивает в цепочку всё, что реализует `Layer`:

```rust
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{Layer, Linear, Relu};

fn classify() -> Result<(), Box<dyn std::error::Error>> {
    // 784 (28x28 pixels) -> 128 -> 10 classes; with_dims draws Kaiming-uniform weights
    let fc1 = Linear::with_dims(784, 128, true, DType::Float32);
    let fc2 = Linear::with_dims(128, 10, true, DType::Float32);

    let pixels: Vec<f32> = (0..784).map(|i| i as f32 / 784.0).collect();
    let image = Tensor::from_slice(pixels).try_reshape(&[1, 784])?;   // batch of 1

    let logits = image.sequential(&[&fc1, &Relu, &fc2])?;
    let probs = logits.softmax(-1)?;
    let prediction = logits.argmax(-1)?;                // Int32 indices

    // Two results sharing the logits: one schedule, one run
    Tensor::realize_batch([&probs, &prediction])?;
    println!("{:?}", probs.as_ndarray::<f32>()?);
    println!("{:?}", prediction.as_vec::<i32>()?);
    Ok(())
}
```

```rust
pub trait Layer {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}
```

`realize_batch` принимает итератор по `&Tensor`; общий подграф (логиты)
вычисляется один раз. `Relu` — `Layer` нулевого размера; те же активации есть и
как методы тензора (`relu`, `sigmoid`, `silu`, `gelu`, `softmax`, `log_softmax`),
как и редукции `sum`, `mean`, `max` и `argmax`; все они принимают ось или `()`
для «всех осей».

---

## Модули и чекпойнты

Структура слоя владеет своими параметрами и гиперпараметрами, нужными для
прямого прохода. `#[derive(Module)]` превращает поля в плоский `StateDict`
(`HashMap<String, Tensor>`) с ключами ровно такими, как их называет PyTorch,
поэтому чекпойнт загружается без ручного сопоставления имён:

```rust
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{LayerNorm, Module, StateDict};

#[derive(Clone, Module)]
struct Block {
    intermediate: usize,            // primitives are skipped automatically
    #[module(skip)]                 // a non-primitive that carries no weights
    dtype: DType,
    norm: LayerNorm,                // child module: "norm.weight", "norm.bias"
    #[module(key = "Wi.weight")]    // checkpoint name, dots allowed
    wi: Tensor,
    #[module(key = "Wo.weight")]
    wo: Tensor,
    #[module(optional)]             // written when Some, absent-tolerant on load
    out_bias: Option<Tensor>,
}

fn load(checkpoint: &StateDict) -> Result<Block, Box<dyn std::error::Error>> {
    let mut block = Block {
        intermediate: 3072,
        dtype: DType::Float32,
        norm: LayerNorm::with_dims(768, true, 1e-5, DType::Float32),
        wi: Tensor::zeros(&[3072, 768], DType::Float32),
        wo: Tensor::zeros(&[768, 3072], DType::Float32),
        out_bias: None,
    };
    // Reads "layers.0.norm.weight", "layers.0.Wi.weight", ...
    block.load_state_dict(checkpoint, "layers.0")?;
    // ...and writes them back out under any prefix
    let _round_trip: StateDict = block.state_dict("layers.0");
    Ok(block)
}
```

| Атрибут | Действие |
|---|---|
| `#[module(key = "Wi.weight")]` | Заменяет сегмент ключа, взятый из имени поля (может содержать точки и цифры) |
| `#[module(key = "")]` | Уплощение: ключи поля используют префикс родителя без изменений |
| `#[module(skip)]` | Игнорирует непримитивное поле (конфигурацию, dtype, режим) |
| `#[module(optional)]` | Обязателен для `Option<Tensor>`: сохраняется при `Some`, при загрузке отсутствие ключа допустимо |
| `#[module(optional = "self.has_bias")]` | Ключ обязателен, если предикат истинен, и пропускается в противном случае |

Дочерние модули компонуются через blanket-реализации: `Vec<M>` и `[M; N]`
нумеруют свои элементы как `0.`, `1.`, …; `Option<M>`, `Box<M>` и `(A, B)`
делегируют так же, а для перечислений derive тоже работает. Прямой проход в
`Module` не входит: он находится в `Layer::forward`, если позволяет сигнатура, и
в собственных методах типа в остальных случаях.

Встроенные слои реализуют оба трейта: `new` — для загруженных тензоров,
`with_dims` — для новой инициализации (веса Kaiming-uniform и нулевые смещения;
тождественное аффинное преобразование для нормализаций):

| Слой | `with_dims` | Ключи state dict |
|---|---|---|
| `Linear` | `(in, out, bias, dtype)` | `weight`, `bias` (если есть) |
| `Conv1d` | `(in_c, out_c, kernel, bias, dtype)` | `weight`, `bias` |
| `Conv2d` / `ConvTranspose2d` | `(in_c, out_c, (kh, kw), bias, dtype)` | `weight`, `bias` |
| `BatchNorm2d` | `(channels, eps, dtype)` | `weight`, `bias`, `running_mean`, `running_var` |
| `LayerNorm` | `(size, bias, eps, dtype)` | `weight`, `bias` (если есть) |
| `RmsNorm` | `(size, eps, dtype)` | `weight` |
| `Embedding` | `(vocab_size, embed_dim, dtype)` | `weight` |

Гиперпараметры задаются методами структуры в стиле builder —
`Conv1d::new(w, bias).with_stride(2).with_padding((1, 1)).with_groups(4)`,
`LayerNorm::with_dims(..).with_axis(-2)`. Пулинг, групповая нормализация и
dropout — это методы тензора (`max_pool2d`, `avg_pool2d`, `group_norm`,
`dropout`), а не структуры.

Чекпойнты загружает `svod-model`: он читает safetensors в том виде, в каком они
сохранены (f32, f16, bf16, fp8, …), и приводит типы только по запросу:

```rust
use std::path::Path;
use svod_model::state::{cast_all, load_safetensors, load_safetensors_dir};
use svod_tensor::nn::Module;

let sd = load_safetensors(Path::new("model.safetensors"))?;      // one file
let sd = load_safetensors_dir(Path::new("checkpoint/"))?;        // or the shards in model.safetensors.index.json
let sd = cast_all(&sd, DType::Float16);
block.load_state_dict(&sd, "layers.0")?;
```

Загрузчики с хаба (`ResNet::from_hub`, `GigaAm::from_hub_with_revision`, …) —
это тот же приём плюс загрузка с Hugging Face; см. [Запуск моделей](./models).

---

## Скомпилировать один раз, запускать многократно

`realize()` планирует и компилирует граф при каждом вызове. Модели, которая
прогоняет один и тот же граф на новых данных (а это любой сервер инференса),
следует скомпилировать его один раз через `prepare()` и повторно запускать план:

```rust
use svod_tensor::Tensor;

fn stream(frames: &[Vec<f32>]) -> Result<(), Box<dyn std::error::Error>> {
    let input = Tensor::from_slice(vec![0.0f32; 1024]);    // owns a host-mappable buffer
    let energy = input.try_mul(&input)?.mean(())?;

    let plan = energy.prepare()?;                          // schedule + compile, once
    for frame in frames {
        input.array_view_mut::<f32>()?.as_slice_mut().unwrap().copy_from_slice(frame);
        plan.execute()?;                                   // replay: no tracing, no compilation
        println!("{}", energy.item::<f32>()?);
    }
    Ok(())
}
```

`prepare()` связывает `energy` с выходным буфером плана, поэтому тензор уже до
первого `execute()` сообщает `realized: true`; читайте его только после
выполнения. `Tensor::prepare_batch([&a, &b])` компилирует несколько выходов в
один план, а `prepare_with(&PrepareConfig)` принимает явную конфигурацию
(`prepare()` использует `PrepareConfig::from_env()`; `PrepareConfig::device_local()`
оставляет выходы на устройстве без отображения на хост).

Повторно запускаемый план не может двух вещей: изменить форму и изменить
значение, которое было свёрнуто в граф при его построении. Для переменного
размера батча или длины последовательности слой моделей предоставляет
`jit_wrapper!`: он объявляет символьные границы, выделяет входные буферы и
перепривязывает переменные при каждом вызове — см.
[JIT-графы](./architecture/jit-graphs). Для ONNX-модели ту же задачу решает
`dim_bindings` при импорте.

---

## Устройства

Тензоры создаются на устройстве по умолчанию: `SVOD_DEVICE`, если переменная
задана, иначе `METAL:0` на macOS и `CPU` на остальных системах. Формат —
`NAME[:index]`, без учёта регистра:

| `SVOD_DEVICE` | Бэкенд |
|---|---|
| `CPU` | LLVM IR, компилируемый внутри процесса (`SVOD_CPU_BACKEND=clang` выбирает C-бэкенд) |
| `CUDA:0` (псевдоним `GPU`) | NVIDIA, `libcuda.so.1` загружается во время выполнения |
| `AMD:0` (псевдоним `HIP`) | AMD, прямые очереди KFD |
| `METAL:0` | GPU Apple |

```rust
use svod_dtype::DeviceSpec;
use svod_tensor::{Tensor, set_default_device, with_default_device};

let on_gpu = cpu_tensor.to(DeviceSpec::Cuda { device_id: 0 });   // lazy COPY node
set_default_device(DeviceSpec::Cuda { device_id: 0 });            // this thread, from now on
with_default_device(DeviceSpec::Cpu, || Tensor::zeros(&[4], DType::Float32));  // scoped
```

Скомпилированные ядра кешируются на диске (`~/.cache/svod/objects` или
`$SVOD_OBJECT_CACHE_DIR`; `SVOD_OBJECT_CACHE=0` отключает кеш), поэтому при
повторном запуске процесса компиляция пропускается. `SVOD_THREADS` ограничивает
пул потоков компиляции и выполнения на CPU, `BEAM=N` включает
[поиск ядер](./architecture/optimizations/kernel-search), а `SVOD_NOOPT=1`
отключает оптимизатор для поиска регрессий бисекцией.

---

## Как это устроено внутри

Граф за тензором — это дерево `UOp`:

```rust
let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
let c = (&a + &b)?;
println!("{}", c.uop().tree());
```

```text
[8592] Add : Scalar(Float32) shape=[Const(3)]
├── [8590] BUFFER(slot=34, addrspace=Some(Global)) : Scalar(Float32) shape=[Const(3)]
│   └── [882] CONST(Int(3)) : Scalar(WeakInt) shape=[]
└── [8591] BUFFER(slot=35, addrspace=Some(Global)) : Scalar(Float32) shape=[Const(3)]
    └── [882] → (see above)
```

Это граф до планирования: узлы `BUFFER` обозначают два входа, а `Add` —
операцию; загрузки, сохранения и диапазоны появляются только после того, как
планировщик превратит граф в ядра. Чтобы их увидеть, подготовьте план и выведите
его ядра:

```rust
let plan = c.prepare()?;
for kernel in plan.kernels() {
    println!("{}\n{}", kernel.entry_point, kernel.code);   // one fused kernel, LLVM IR on the CPU
}
```

`SVOD_DUMP_LLVM_IR=<dir>` записывает IR каждого ядра в `<dir>/<name>.ll` без
изменений в коде, а `SVOD_DUMP_LINEAR=<dir>` выгружает линеаризованную
UOp-программу. Страница [конвейер выполнения](./architecture/pipeline)
прослеживает одно ядро через все этапы.

---

## Рекуррентные слои

`rnn()`, `gru()` и `lstm()` — builder-методы `Tensor`. Они принимают как имена
весов PyTorch (`weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`, `h0`, `c0`), так и
имена ONNX (`w`, `r` — в `gru()` он называется `r_weights` — `bias`,
`initial_h`, `initial_c`) и сами переупорядочивают блоки гейтов:

```rust
use ndarray::Array3;
use svod_tensor::Tensor;

// seq=2, batch=1, input=3, hidden=4
let x = Tensor::from_ndarray(&Array3::from_elem((2, 1, 3), 0.1f32));
let w = Tensor::from_ndarray(&Array3::from_elem((1, 12, 3), 0.1f32));
let r = Tensor::from_ndarray(&Array3::from_elem((1, 12, 4), 0.1f32));

let out = x.gru().w(&w).r_weights(&r).hidden_size(4).call()?;
// ONNX-shaped: y [seq, num_directions, batch, hidden], y_h [num_directions, batch, hidden]
// PyTorch-shaped: output [seq, batch, D*hidden], h_n [num_directions, batch, hidden]
assert_eq!(out.y.dims()?, vec![2, 1, 1, 4]);
assert_eq!(out.output.dims()?, vec![2, 1, 4]);
```

`layout` выбирает `RnnLayout::SeqFirst` (`[seq, batch, input]`, по умолчанию)
или `BatchFirst`; `direction` принимает `RnnDirection::{Forward, Backward, Bidirectional}`, и двунаправленный проход конкатенирует два направления по оси
признаков. `linear_before_reset` у GRU по умолчанию следует размещению PyTorch
при весах PyTorch и размещению ONNX при весах ONNX. `LstmOutput` добавляет
`y_c` / `c_n` для состояния ячейки.

Ось времени должна быть конкретной, а ось батча может быть символьной. Для
написанного вручную цикла — например, декодера, который делает шаг по одному
токену, — используйте ячейки напрямую: `RnnCell`/`GruCell` предоставляют
`step(&x, &h) -> Result<Tensor>`, `LstmCell` — `step(&x, &h, &c) -> Result<(Tensor, Tensor)>`,
а `RnnStack::new(cells)` делает шаг сразу по всему стеку.

---

## Спектрограммы

`stft()` — это один `conv1d` с ядром оконного ДПФ, поэтому всё преобразование
остаётся в графе (и ось батча может оставаться символьной). Результат имеет
форму `[B, F, T, 2]` — или `[F, T, 2]` для сигнала `[L]` без батча — с
`(real, imag)` на последней оси, что соответствует
`torch.stft(..., return_complex=false)`:

```rust
use svod_tensor::Tensor;
use svod_tensor::nn::Window;

let x = Tensor::from_slice(vec![0.25f32; 64]);
let spec = x.stft().n_fft(16).hop(4).window(Window::Hann).call()?;
assert_eq!(spec.dims()?, vec![9, 17, 2]);   // [F, T, (re, im)]

let mag = spec.magnitude(0.0)?;             // sqrt(re² + im² + eps)
let signal = spec.istft().n_fft(16).hop(4).window(Window::Hann).length(64).call()?;
```

Значения по умолчанию совпадают с torch: `hop = n_fft / 4`, `win_length = n_fft`,
периодическое окно Ханна, `center` (дополнение отражением), `onesided`, без
нормализации — и `istft` нужно передать те же параметры. `Window` — это `Hann`,
`Hamming`, `Povey`, `Rectangular` или `Custom(tensor)`, а
`Tensor::window(&Window::Hann, n, periodic, dtype)` создаёт окно в виде тензора.
Помимо `magnitude`, для последней оси размера 2 есть `power`, `complex_abs`,
`complex_mul` и `Tensor::complex_from_polar(&mag, &phase)`.

Мел-фронтенд — это тот же граф со свёрткой с банком фильтров и логарифмом в
конце. `mel_spectrogram()` принимает параметры кадрирования `stft` и
мел-параметры и возвращает `[B, n_mels, T]` (`[n_mels, T]` без батча):

```rust
use svod_tensor::Tensor;
use svod_tensor::nn::{MelLog, MelNorm, MelScale};

let x = Tensor::from_slice(vec![0.25f32; 16000]);
let mel = x
    .mel_spectrogram()
    .sample_rate(16000)
    .n_fft(400)
    .hop(160)
    .n_mels(80)
    .mel_scale(MelScale::Slaney)
    .norm(MelNorm::Slaney)
    .log(MelLog::Whisper)
    .call()?;
assert_eq!(mel.dims()?, vec![80, 101]);
```

Значения по умолчанию взяты из `MelSpectrogram` в torchaudio (шкала HTK, без
нормализации, `power = 2`, `f_min = 0`, `f_max = sample_rate / 2`, без
логарифма); `MelScale::Slaney` вместе с `MelNorm::Slaney` дают
`librosa.filters.mel` — банк фильтров, на котором построен Whisper.
`MelLog::Ln { min, max }` — это `ln(clamp(x))`, а `MelLog::Whisper` — хвост
`log_mel_spectrogram`: `log10` / ограничение снизу уровнем `max - 8` /
`(x + 4) / 4`; `mel_log` применяет любой из них отдельно, `preemphasis` и
`remove_dc` покрывают фронтенды в стиле Kaldi, а `filterbank(&t)` подставляет
заранее вычисленную таблицу `[n_mels, F]` (её строит `Tensor::mel_filterbank(...)`).

---

## Ошибки

Каждый метод тензора, который может завершиться ошибкой, возвращает
`svod_tensor::error::Result<T>`; его ошибка — `Error(Box<ErrorKind>)` размером
с указатель. Сопоставляйте причину через `err.kind()` (или `into_kind()`, чтобы
получить её по значению). Зависимые крейты преобразуют её через
`context(false)` из snafu, поэтому собственное перечисление ошибок модели
поглощает её простым `?` — без `.context(TensorSnafu)` в каждом месте вызова.

Ошибкой может завершиться не всё. `cast`, `neg`, `abs`, `floor`, `ceil`, `round`,
`trunc`, `square`, `sign` и конструкторы `Tensor::full` / `zeros` / `ones` не
могут завершиться ошибкой и возвращают обычный `Tensor`; `-&a` тоже возвращает
обычный тензор, а бинарные операторы — `Result<Tensor>`.

---

## Итоги

| Задача | Код |
|---|---|
| Создать тензор | `Tensor::from_slice([1.0f32, 2.0])`, `Tensor::from_ndarray(&arr)` |
| Арифметика | `(&a + &b)?`, `(&a * 2.0)?`, `(2.0f32 * &a)?`, `-&a` |
| Изменить форму | `t.try_reshape(&[2, 3])?` |
| Транспонировать | `t.try_transpose(0, 1)?` |
| Умножить матрицы | `a.dot(&b)?` |
| Посмотреть форму | `t.dims()?`, `t.dim_const(-1)?`, `t.dtype()` |
| Линейный слой | `Linear::with_dims(in, out, bias, dtype)` |
| Цепочка слоёв | `x.sequential(&[&fc1, &Relu, &fc2])?` |
| Активация | `t.relu()?`, `t.softmax(-1)?` |
| Загрузить веса | `model.load_state_dict(&sd, "")?` |
| Спектрограмма | `x.stft().n_fft(512).hop(160).call()?` |
| Мел-спектрограмма | `x.mel_spectrogram().sample_rate(16000).n_fft(400).n_mels(80).call()?` |
| Рекуррентный слой | `x.lstm().weight_ih(&w).weight_hh(&r).hidden_size(h).call()?` |
| Выполнить | `t.realize()?` |
| Выполнить несколько тензоров | `Tensor::realize_batch([&a, &b])?` |
| Скомпилировать один раз | `let plan = t.prepare()?; plan.execute()?` |
| Извлечь данные | `t.to_vec::<f32>()?`, `t.to_ndarray::<f32>()?`, `t.item::<f32>()?` |
| Выбрать устройство | `SVOD_DEVICE=CUDA:0`, `t.to(DeviceSpec::Cuda { device_id: 0 })` |

**Что дальше:**

- [Запуск моделей](./models) — готовые модели для речи, текста и зрения
- [ONNX-инференс](./onnx) — импорт файла `.onnx` в тот же граф
- [JIT-графы](./architecture/jit-graphs) — `jit_wrapper!`, символьные батчи и состояние на устройстве
- [Конвейер выполнения](./architecture/pipeline) — как граф превращается в ядра
