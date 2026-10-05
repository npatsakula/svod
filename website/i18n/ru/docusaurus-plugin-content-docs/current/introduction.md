---
sidebar_label: Введение
---

<div align="center">

# Svod

**Компилятор глубокого обучения и движок инференса на Rust.**

[![CI](https://github.com/npatsakula/svod/actions/workflows/ci.yml/badge.svg)](https://github.com/npatsakula/svod/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-svod.vpermilp.online-blue)](https://svod.vpermilp.online/docs/introduction)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/npatsakula/svod/tree/main/LICENSE)

[Документация](https://svod.vpermilp.online/docs/introduction) ·
[Модели](https://github.com/npatsakula/svod/tree/main/#models-and-pipelines) ·
[Архитектура](https://svod.vpermilp.online/docs/architecture/pipeline) ·
[Доклады](https://github.com/npatsakula/svod/tree/main/#talks-and-writing) ·
[Планы](https://github.com/npatsakula/svod/tree/main/#roadmap)

</div>

Svod компилирует ленивые тензорные графы в слитые (fused) ядра для CPU, GPU AMD и
NVIDIA без единого вендорского рантайма в цепочке: ни PyTorch, ни ROCm/HIP, ни
CUDA toolkit. Он следует дизайну [Tinygrad](https://github.com/tinygrad/tinygrad):
небольшое проверяемое IR (UOp), перезаписи на паттернах и прямой конвейер от
тензоров до машинного кода.

Svod поставляется с речевыми, текстовыми и визуальными моделями, сверенными со
своими PyTorch-референсами, плюс импортёр ONNX для всего остального.

## Почему Svod

- **Одно представление от начала до конца.** PyTorch проводит модель через семь
  и более IR, и каждая граница — это мост, который вендор должен построить, и
  разрыв контекста при отладке. В Svod единый граф UOp строится, оптимизируется,
  планируется и рендерится. Новому ускорителю нужны лишь три части: генератор
  кода, аллокатор буферов и запускатель ядер.
- **Промышленный Rust.** Статические типы и нативная конкурентность вместо
  Python-обвязки вокруг GIL. Деплой — один бинарник, без привязок к LibTorch или
  ONNX Runtime и без вендорских SDK.
- **Знакомый API.** Тензорный API повторяет PyTorch вплоть до именованных
  аргументов, поэтому порт модели читается как референс, а сам перенос в
  основном механический.
- **Скорость там, где она важна.** Когда компилятора недостаточно, tile-DSL `tk`
  даёт рукописные ядра в том же IR. Они остаются видимыми профилировщику и
  трекеру происхождения, а не живут где-то сбоку непрозрачными бинарниками.

Мотивация и дизайн описаны в статье
[Developing Svod](https://blog.vpermilp.online/en/blog/svod-intro/).

## Модели и пайплайны

| Область | Модели |
|---|---|
| Распознавание речи | Whisper, GigaAM v3 (CTC, RN-T) |
| Детекция речи (VAD) | FireRedVAD, Silero VAD |
| Улучшение речи | GTCRN |
| Анализ дикторов | DiariZen, WeSpeaker |
| Текстовые эмбеддинги и реранкинг | BGE-M3, Qwen3-Embedding, ModernBERT |
| Зрение | YOLO26, ResNet |
| Всё остальное | [Импортёр ONNX](https://github.com/npatsakula/svod/tree/main/onnx/) ([покрытие операторов](https://github.com/npatsakula/svod/tree/main/onnx/PARITY.md)) |

Веса берутся напрямую из Hugging Face Hub, а выходы сверяются с референсными
реализациями. В [`model/`](https://github.com/npatsakula/svod/tree/main/model/) перечислены варианты, ссылки на
исходные модели и запускаемые примеры; [`arch`](https://github.com/npatsakula/svod/tree/main/arch/) содержит декодеры и пайплайн для длинного аудио.

## Движок

### Захват графа: компиляция один раз, воспроизведение много раз

Модель трассируется один раз в план выполнения, и каждый следующий вызов лишь
воспроизводит его. Символьные размерности (батч, длина последовательности)
связываются на каждом вызове без перекомпиляции. Рекуррентное состояние остаётся
на устройстве между вызовами, а планировщик памяти переиспользует промежуточные
буферы через TLSF-арену. Статические цепочки воспроизводятся как один аппаратный
граф: **CUDA Graphs** на NVIDIA, AQL/PM4-граф с одним doorbell на воспроизведение
на AMD (как в HCQGraph) и indirect command buffers на Metal.
См. [JIT-графы](https://svod.vpermilp.online/docs/architecture/jit-graphs).

```rust
jit_wrapper! {
    GigaAmEncoderJit(GigaAm) {
        mel: Tensor,
        lengths: Tensor,

        outputs { frames },

        build(mel, lengths) {
            model.encoder.forward_batch(mel, lengths)
        }
    }
}
// let mut jit = GigaAmEncoderJit::new(model);
// jit.prepare(..)?;   // trace, schedule and compile once
// jit.execute()?;     // replay on every chunk
```

### Перезаписи, доказанные Z3

Каждая оптимизация — декларативная перезапись в DSL `patterns!`. Алгебраические
и индексные упрощения проверяются **SMT-решателем Z3**: он доказывает, что
переписанное выражение равно исходному для любого входа, либо возвращает
контрпример. Остальную часть конвейера покрывают property-based тесты.
См. [Система паттернов](https://svod.vpermilp.online/docs/architecture/optimizations/pattern-system).

### Платформенно-специфичная генерация кода

- **Тензорные ядра** выбираются по архитектуре: NVIDIA sm_75/80/89, AMD RDNA3,
  RDNA4 и CDNA3/4, а также Apple Metal. fp8 доступен на sm_89 и CDNA3.
- **Tile-ядра (`tk`)**: tile-DSL в стиле ThunderKittens на Rust для GEMM,
  flash attention, RMSNorm и k-means. Один исходник ядра опускается в AMD
  MFMA/WMMA (gfx942, gfx11, gfx12), CUDA `mma.sync` (sm_80+) и Apple
  `simdgroup_matrix` (Apple7+). Формы тайлов
  автоматически подбираются при первом использовании и кэшируются. См.
  [Tile-ядра](https://svod.vpermilp.online/docs/tile-kernels/overview).

  ```rust
  fn micro_matmul(ker: &Kernel) -> Arc<UOp> {
      let w = ker.warp();
      let a = ker.rt((64, 64), DType::BFloat16, Row, RT_16X16);
      let b = ker.rt((64, 64), DType::BFloat16, Col, RT_16X16);
      let c = ker.rt((64, 64), DType::Float32, Col, RT_16X16);
      let out = w.mma_ab(w.zero(c), &a, &b); // one matrix-core instruction per fragment
      ker.finish(1)
  }
  ```
- **Поиск ядер**: рукописные эвристики или BEAM-поиск по пространству
  оптимизаций с постоянным кэшем на диске. См.
  [Поиск ядер](https://svod.vpermilp.online/docs/architecture/optimizations/kernel-search).
- **CPU**: векторизованный LLVM IR, компилируемый внутри процесса, собственный
  загрузчик ELF для x86_64, aarch64, riscv64, loongarch64 и ppc64le, а также
  многопоточные ядра.

### Пути данных без копирования

Инициализаторы ONNX и тензоры `Tensor::from_path` лениво отображаются в память с диска. Буферы устройства
поддерживают под-представления (sub-views). Хост-код читает и пишет реализованные тензоры через заимствованные
представления `ndarray` (`array_view`, `array_view_mut`), так что подача данных в захваченный план
ничего не копирует.

### Слияние ядер и атрибуция

Планировщик RANGEIFY сливает поэлементные операции, редукции и операции
перемещения в как можно меньшее число ядер. Каждое ядро запоминает, откуда оно
взялось (путь модуля, узел ONNX или строка исходника), поэтому профилировщик
может отнести время устройства, roofline GFLOP/s и GB/s, занятость и аппаратные
счётчики (AMD SQ, NVIDIA CUPTI) обратно к коду модели. См.
[Происхождение ядер](https://svod.vpermilp.online/docs/architecture/kernel-origins).

## Бэкенды

| Устройство | Селектор | Компиляция | Рантайм |
|---|---|---|---|
| CPU | `CPU` (по умолчанию вне macOS) | LLVM IR через `libLLVM`, загружаемый во время выполнения, с откатом на `clang`; C-бэкенд Clang | Собственный ELF-загрузчик JIT, многопоточный |
| GPU AMD | `AMD:N` | `clang --target=amdgcn-amd-amdhsa` | Прямые очереди KFD (AQL/PM4), без HIP и рантайма ROCm |
| GPU NVIDIA | `CUDA:N` | `clang` NVPTX → PTX → `ptxas` или JIT драйвера | `libcuda.so.1`, загружаемая во время выполнения, без CUDA toolkit |
| GPU Apple | `METAL:N` (по умолчанию на macOS) | MSL → metallib | Фреймворки Metal, загружаемые во время выполнения |

Каждый GPU-бэкенд вкомпилирован и регистрируется только при наличии
оборудования. Чтобы выбрать бэкенд, задайте `SVOD_DEVICE` или вызовите
`Tensor::to(device)`.

CPU-код тестируется на x86_64, aarch64, riscv64 и ppc64le под Linux и macOS.
GPU: AMD RDNA 3.5, RDNA 4 и CDNA 3, NVIDIA sm_80 и новее, а также Apple M3 и новее.

## Доклады и статьи

[Developing Svod](https://blog.vpermilp.online/en/blog/svod-intro/) — статья в
блоге (август 2026) о том, зачем нужен Svod, его архитектуре и планах.

| Событие | Доклад | Язык |
|---|---|---|
| [Data Fest 2026](https://ods.ai/events/df2026-31-may-online) (онлайн, 31 мая 2026) | Пишем самый быстрый инференс Sber GigaAM на Svod | Русский |
| [RustCon 2025](https://rustcon.ru/morok-minimalistichnyy-deep-learning-freymvork-na-rust) (Москва, ноябрь 2025) | Morok: минималистичный deep-learning-фреймворк на Rust | Русский |
| [Stereo Data Ёлка 2025](https://ods.ai/events/data-elka-2025-vk-offline-spb) (Санкт-Петербург, январь 2026) | ML на Rust | Русский |

Morok — прежнее название Svod.

## Рабочее пространство

| Крейт | Роль |
|---|---|
| [`dtype`](https://github.com/npatsakula/svod/tree/main/dtype/) | Скалярные, векторные, указательные и image-типы, включая bf16 и fp8 |
| [`ir`](https://github.com/npatsakula/svod/tree/main/ir/) | Графовое IR UOp с хеш-консингом, символьными целыми и происхождением |
| [`macros`](https://github.com/npatsakula/svod/tree/main/macros/) | DSL перезаписей `patterns!` и `jit_wrapper!` |
| [`schedule`](https://github.com/npatsakula/svod/tree/main/schedule/) | RANGEIFY, проходы перезаписи, эвристики и BEAM, верификация Z3 |
| [`codegen`](https://github.com/npatsakula/svod/tree/main/codegen/) | Рендереры LLVM IR (CPU, AMDGPU, NVPTX), C и MSL |
| [`device`](https://github.com/npatsakula/svod/tree/main/device/) | Буферы, аллокаторы, mmap, драйверы KFD, CUDA и Metal, аппаратные графы |
| [`runtime`](https://github.com/npatsakula/svod/tree/main/runtime/) | Компиляция ядер, кэширование, планы выполнения и профилировщик |
| [`tensor`](https://github.com/npatsakula/svod/tree/main/tensor/) | Ленивый тензорный API, модули `nn` и планировщик памяти |
| [`tk`](https://github.com/npatsakula/svod/tree/main/tk/) | Tile-DSL ядер и библиотека ядер |
| [`onnx`](https://github.com/npatsakula/svod/tree/main/onnx/) | Импортёр ONNX |
| [`arch`](https://github.com/npatsakula/svod/tree/main/arch/) | Хост-декодеры, VAD-сегментация и аудио-пайплайны |
| [`model`](https://github.com/npatsakula/svod/tree/main/model/) | Предобученные модели и примеры |

## Использование библиотеки

Модели собираются в пайплайны. Вот распознавание длинной русской речи моделью
GigaAM с сегментацией через FireRedVAD:

```rust
let model = GigaAm::from_hub_with_revision("vpermilp/GigaAM-v3", "ctc")?;
let bounds = EncoderBounds {
    sample_rate: model.config.sample_rate as u32,
    hop_length: model.config.hop_length,
    subsampling_factor: model.config.subsampling_factor,
    max_mel_frames: model.config.max_mel_frames,
    recommended_target_secs: model.recommended_chunk_secs(),
};
let splitter = FireRedVadSplitter::from_hub(&bounds)?;
let mut asr = Asr::assemble(splitter, |max_chunk| GigaAmTranscriber::new(model, opts, max_chunk))?;
let result = asr.transcribe_default(&waveform)?;
```

Любую ONNX-модель можно скомпилировать один раз и воспроизводить. Граф трассируется на вашем
входном тензоре, поэтому каждое воспроизведение видит новые данные, записанные в него:

```rust
let proto = ModelProto::decode(std::fs::read("model.onnx")?.as_slice())?;
let input = Tensor::from_ndarray(&first_batch); // [1, 3, 224, 224] f32

let OnnxModel { outputs, .. } = OnnxImporter::new().import_model_with_inputs(
    proto,
    HashMap::from([("input".to_string(), input.clone())]),
    &[("batch", 1)],
)?;

let plan = Tensor::prepare_batch(outputs.values())?; // compile once
plan.execute()?;

for batch in batches {
    input.array_view_mut::<f32>()?.as_slice_mut().unwrap().copy_from_slice(&batch);
    plan.execute()?; // replay: no tracing, no compilation, no allocation
}
```

## Сборка {#building}

Nix flake фиксирует версии всех компиляторов и библиотек, и CI использует тот
же flake:

```bash
nix develop      # development shell
nix flake check  # the CI suite: clippy, nextest (with Z3 and proptest), fmt
```

Без Nix потребуется следующее:

| Зависимость | Версия | Обязательна | Назначение |
|---|---|---|---|
| Rust | 1.88+ | да | Edition 2024 |
| LLVM | ≥ 16 | да | Генерация кода для CPU; `libLLVM` загружается во время выполнения |
| Clang | — | да | Компиляция GPU-ядер, C-бэкенд, запасной вариант при отсутствии `libLLVM` |
| protobuf, pkgconf, zlib, libffi, libxml2 | — | да | Протоколы ONNX и тулчейн LLVM |
| Z3 | ≥ 4.15 | нет | Верификация перезаписей (`--features z3`) |
| Драйвер NVIDIA | CUDA ≥ 12.0 (R525) | нет | Бэкенд CUDA |
| Драйвер ядра amdgpu (KFD) | — | нет | Бэкенд AMD |

```bash
cargo test --workspace
cargo test --workspace --features z3,proptest
```

`SVOD_THREADS` задаёт единый бюджет потоков, используемый для компиляции ядер и
выполнения CPU-ядер.

## Планы

- **AOT-компиляция:** сериализация оптимизированных графов и скомпилированных
  ядер, чтобы модель стартовала мгновенно и работала там, где компилятора нет
  (например, WASM).
- **Примитивы анализа данных:** GPU-ядра в стиле FlashAttention для k-means,
  kNN, PCA, SVD, (H)DBSCAN, UMAP и t-SNE на всех бэкендах. k-means и kNN уже
  есть в `tk`.
- **Формальная верификация сгенерированного кода:** аннотированный C-вывод,
  доказывающий отсутствие выходов за границы массивов и lossy-приведений.
- **Больше оборудования:** серверные (MI300–MI450, H100–B200), потребительские
  (Ryzen AI, Apple M3–M5, RTX 30–50) и встраиваемые (Snapdragon X, RK3588)
  целевые платформы за одним тензорным API, плюс userspace-драйвер AMD без
  зависимости от программного стека AMD.

## Лицензия

[MIT](https://github.com/npatsakula/svod/tree/main/LICENSE)
