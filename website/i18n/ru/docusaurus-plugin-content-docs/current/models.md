---
sidebar_label: Запуск моделей
---

# Запуск моделей

`svod-model` содержит порты предобученных моделей на чистом Rust. Каждая модель
при первом использовании скачивает веса с Hugging Face Hub, один раз
компилирует свой граф через `jit_wrapper!` и сверяется с исходной реализацией
на PyTorch в тестах крейта. `svod-arch` добавляет вокруг них инфраструктуру на
стороне хоста: CTC- и RN-T-декодеры, нарезку на фрагменты по VAD и конвейер для
длинных аудиозаписей.

| Область | Модель | Модуль | Веса |
|---|---|---|---|
| Распознавание речи | GigaAM v3 (CTC, RN-T) | `gigaam` | `vpermilp/GigaAM-v3` |
| Распознавание речи | Whisper (tiny … large-v3, turbo) | `whisper` | `vpermilp/whisper` |
| Детекция речевой активности | FireRedVAD (пакетный, потоковый) | `firered_vad` | `vpermilp/firered_vad` |
| Детекция речевой активности | Silero VAD 16k | `silero_vad` | `vpermilp/silero-vad` |
| Улучшение речи | GTCRN | `gtcrn` | `vpermilp/gtcrn` |
| Диаризация дикторов | DiariZen (WavLM + Conformer) | `diarizen` | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` |
| Эмбеддинги дикторов | WeSpeaker ResNet34 | `wespeaker` | `pyannote/wespeaker-voxceleb-resnet34-LM` |
| Текстовые эмбеддинги | BGE-M3 (dense, sparse, ColBERT), BGE-reranker-v2-m3 | `bgem3` | `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3` |
| Текстовые эмбеддинги | Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B | `qwen3` | `Qwen/Qwen3-Embedding-0.6B` |
| Текстовые эмбеддинги, fill-mask | ModernBERT base / large | `modernbert` | `answerdotai/ModernBERT-base` |
| Компьютерное зрение | ResNet 18 / 34 / 50 / 101 / 152 | `resnet` | `timm/resnet*.a1_in1k` |
| Компьютерное зрение | YOLO26 (detect, cls, seg, obb, pose, depth, semseg) | `yolo` | `ultralytics/yolo26n` … `yolo26x` |

В [README крейта моделей](https://github.com/npatsakula/svod/tree/main/model)
перечислены исходные репозитории и тесты соответствия для каждой строки.

---

## Подготовка

```toml
[dependencies]
svod-model  = "0.1"
svod-arch   = "0.1"   # Asr, splitters, decoders
svod-tensor = "0.1"
svod-dtype  = "0.1"
```

Для сборки на машине нужны LLVM и Clang (см.
[введение](./introduction#building)); для CPU больше ничего не требуется.
GPU-бэкенды встроены в сборку и регистрируются сами, если установлен драйвер, —
выберите бэкенд через `SVOD_DEVICE`:

```bash
SVOD_DEVICE=CUDA:0 cargo run --release -p svod-model --example whisper_infer -- audio.wav
SVOD_DEVICE=AMD:0  cargo run --release -p svod-model --example gigaam_infer -- audio.wav
```

Веса сохраняются в кеш Hugging Face (`HF_HUB_CACHE`, по умолчанию
`~/.cache/huggingface`; `HF_TOKEN` для репозиториев с ограниченным доступом).
Скомпилированные ядра кешируются в `~/.cache/svod/objects`, поэтому при
повторном запуске процесса компиляция пропускается.

---

## Примеры

Каждый пример — законченная программа в
[`model/examples`](https://github.com/npatsakula/svod/tree/main/model/examples)
и лучшая отправная точка для соответствующей модели. Все аудиопримеры ожидают
моно-WAV с частотой 16 кГц.

```bash
cargo run -p svod-model --release --example gigaam_infer   -- audio.wav [--rnnt] [--timestamps]
cargo run -p svod-model --release --example whisper_infer  -- audio.wav [--size base] [--language auto] [--timestamps]
cargo run -p svod-model --release --example vad_stream     -- audio.wav                # streaming VAD events
cargo run -p svod-model --release --example vad_bench      -- audio.wav                # Silero vs FireRedVAD
cargo run -p svod-model --release --example gtcrn_enhance  -- --in noisy.wav --out clean.wav --hub
cargo run -p svod-model --release --example qwen3_embed    -- --texts texts.txt --batch 8 --max-len 512
cargo run -p svod-model --release --example resnet_classify -- --hub --image dog.bin --side 224
cargo run -p svod-model --release --example yolo_detect    -- --hub --scale small --image photo.bin --side 640
```

Входные изображения — сырые файлы `f32` в формате NCHW (`3 × side × side × 4`
байт, значения нормализованы в диапазон 0–1); если `--image` не указан, примеры
синтезируют тестовый узор, поэтому каждая команда выше работает без
дополнительных файлов.

`--profile` в примерах для речи и эмбеддингов выводит отчёт по этапам со
временем на устройстве, а также GFLOP/s и GB/s по roofline-модели для каждого
ядра. С `SVOD_ORIGIN=1` каждое ядро атрибутируется коду модели, который его
породил, а `--origin-depth N` сворачивает отчёт до глубины модулей `N`:

```bash
SVOD_ORIGIN=1 cargo run -p svod-model --release --example gigaam_infer -- \
    audio.wav --profile --origin-depth 3 --profile-json profile.json
```

---

## Распознавание речи

### GigaAM

Транскрибация длинных записей — это конвейер: VAD разбивает сигнал на речевые
фрагменты под размер энкодера, каждый фрагмент транскрибируется, а части
сшиваются обратно с временными метками. `Asr::assemble` связывает обе половины
так, чтобы JIT транскрайбера был подготовлен один раз под размер фрагмента,
который может выдать сплиттер:

```rust
use svod_arch::pipelines::audio::{Asr, RunOptions};
use svod_model::audio::EncoderBounds;
use svod_model::firered_vad::FireRedVadSplitter;
use svod_model::gigaam::{GigaAm, GigaAmTranscriber, TranscribeOpts};

let model = GigaAm::from_hub_with_revision("vpermilp/GigaAM-v3", "ctc")?;   // "e2e_rnnt" for RN-T
let bounds = EncoderBounds {
    sample_rate: model.config.sample_rate as u32,
    hop_length: model.config.hop_length,
    subsampling_factor: model.config.subsampling_factor,
    max_mel_frames: model.config.max_mel_frames,
    recommended_target_secs: model.recommended_chunk_secs(),
};
let splitter = FireRedVadSplitter::from_hub(&bounds)?;
let opts = TranscribeOpts::default();
let mut asr = Asr::assemble(splitter, |max_chunk| GigaAmTranscriber::new(model, opts, max_chunk))?;

let result = asr.transcribe(&waveform, RunOptions { words: true, ..Default::default() })?;
println!("{}", result.text);
for chunk in &result.chunks {
    for word in chunk.words.iter().flatten() {
        println!("[{:.2} - {:.2}] {}", chunk.start_sec + word.start, chunk.start_sec + word.end, word.text);
    }
}
```

Голова (CTC или RN-T) определяется ревизией весов. `TranscribeOpts` содержит
структурные параметры — `beam_decode` и бюджет на матрицу scores в SDPA
`max_scores_mib`, — а `RunOptions` (`words`, `segments`, `profile`)
переключается при каждом вызове, поэтому один `Asr` обслуживает все режимы.
`FireRedVadSplitter::builder()` открывает параметры нарезки (`threshold`,
`min_duration`, `max_duration`, `target_duration`, …); `SVOD_VAD_THRESHOLD` и
`SVOD_VAD_TARGET_CHUNK_SECS` переопределяют значения по умолчанию из окружения.
`FixedLengthSplitter` или `SileroVadSplitter` подставляются без изменений в
транскрайбере.

### Whisper

Whisper декодирует фиксированные 30-секундные окна, поэтому сплиттер —
`FixedLengthSplitter`. Транскрайбер выполняет цепочку мел → энкодер →
лучевое декодирование с кешем и откатом по температуре, а затем, по запросу,
выравнивание слов через DTW:

```rust
use svod_arch::pipelines::audio::{Asr, FixedLengthSplitter, RunOptions};
use svod_model::whisper::{
    CHUNK_LENGTH, DecodeOptions, DecodeStrategy, ModelDimensions, SAMPLE_RATE, Whisper,
    WhisperAlignedTranscriber, WhisperSize, WhisperTokenizer,
};

let size = WhisperSize::Base;
let dims = ModelDimensions::for_size(size);
let model = Whisper::from_hub("vpermilp/whisper", size.name(), dims)?;
let tokenizer = WhisperTokenizer::from_hub(model.is_multilingual(), model.dims.num_languages())?;

let options = DecodeOptions {
    language: None,                                   // detect
    strategy: DecodeStrategy::Beam { size: 5 },
    ..Default::default()
};
let window = CHUNK_LENGTH * SAMPLE_RATE;
let transcriber = WhisperAlignedTranscriber::new(model, tokenizer, options, size, window)?;
let mut asr = Asr::new(FixedLengthSplitter::new(window, SAMPLE_RATE), transcriber);

let result = asr.transcribe(&waveform, RunOptions::default())?;
println!("{}", result.text);
```

`WhisperSize::from_name("large-v3")` разбирает написание из CLI, а
`WhisperPlan` (`encoder_batch`, `decoder_slots`, `alignment_batch`) задаёт
размеры скомпилированных графов, если значения по умолчанию не подходят для
устройства.

---

## Детекция речевой активности и улучшение речи

`FireRedVadStreamer` — потоковый фронтенд: подавайте сэмплы по мере
поступления и получайте границы речи в виде событий. Кеши свёрток
переиспользуются на устройстве между подачами данных, поэтому каждые
`chunk_frames` аудио стоят одного небольшого запуска ядра:

```rust
use svod_model::firered_vad::{FireRedVadStreamer, VadEvent};

let mut vad = FireRedVadStreamer::from_hub()?;      // or builder().model(m).chunk_frames(16).build()
for block in waveform.chunks(1600) {                 // 100 ms at 16 kHz
    for event in vad.push(block)? {
        match event {
            VadEvent::SpeechStart { frame } => println!("start {:.2}s", frame as f32 / 100.0),
            VadEvent::SpeechEnd { start_frame, end_frame } => println!("end   {:.2}s", end_frame as f32 / 100.0),
        }
    }
}
let flush = vad.flush()?;                            // closes the last segment
for (start, end) in &flush.timestamps {              // seconds
    println!("{start:.2}s - {end:.2}s");
}
```

Улучшение речи GTCRN — это один JIT для цепочки STFT → сеть → ISTFT.
Рекуррентность GRU разворачивается по кадрам, поэтому пример прогоняет её
фиксированными фрагментами по 128 кадров с одним `prepare`:

```rust
use svod_model::gtcrn::{Gtcrn, GtcrnJit, HOP};
use svod_model::jit::InputSpec;

const CHUNK: usize = 128 * HOP;
let mut jit = GtcrnJit::new(Gtcrn::from_hub()?);
jit.prepare(InputSpec::f32(&[1, CHUNK]))?;

let mut out = vec![0.0f32; CHUNK];
for chunk in waveform.chunks(CHUNK) {               // pad the last one to CHUNK
    jit.waveform_mut()?.copyin(bytemuck::cast_slice(chunk))?;
    jit.execute()?;
    jit.output()?.copyout(bytemuck::cast_slice_mut(&mut out))?;
}
```

---

## Компьютерное зрение

Каждая модель компьютерного зрения — это один `jit_wrapper!`: один вход
`images`, `batch_var`, благодаря которому один план обслуживает любой размер
батча до `max_batch_size`, и один выход. Размер изображения фиксируется при
`prepare`:

```rust
use svod_dtype::DType;
use svod_model::jit::InputSpec;
use svod_model::resnet::{OutputMode, ResNet, ResNetDepth, ResNetJit};

let model = ResNet::from_hub("timm/resnet50.a1_in1k", ResNetDepth::R50,
                             OutputMode::Classification { num_classes: 1000 })?;
let mut jit = ResNetJit::new(model);
jit.prepare(InputSpec::new(&[1, 3, 224, 224], DType::Float32))?;

jit.images_mut()?.copyin(bytemuck::cast_slice(&nchw_pixels))?;   // [1, 3, 224, 224] f32
jit.execute_bound(1)?;                                            // batch = 1
let logits = jit.logits_to_vec::<f32>()?;                         // [1000]
```

Обёртка, стоящая за ним, дословно:

```rust
jit_wrapper! {
    ResNetJit(ResNet) {
        inputs { images: Tensor }
        batch_var b: (1, model.config.max_batch_size),
        outputs { logits }
        build(images) { model.forward(images) }
    }
}
```

YOLO26 устроена так же. Голова детекции возвращает декодированные рамки `xyxy`
с оценками классов, `[B, 4 + nc, anchors]`, а `postprocess_raw` выполняет
отбор top-k:

```rust
use svod_model::yolo::{Yolo26Detect, Yolo26DetectJit, YoloConfig, YoloScale, postprocess_raw};

let cfg = YoloConfig::new(YoloScale::Nano, 80);
let mut jit = Yolo26DetectJit::new(Yolo26Detect::from_hub("ultralytics/yolo26n", cfg)?);
jit.prepare(InputSpec::f32(&[1, 3, 640, 640]))?;

jit.images_mut()?.copyin(bytemuck::cast_slice(&nchw_pixels))?;
jit.execute_bound(1)?;
let detections = postprocess_raw(&jit.predictions_to_vec::<f32>()?, &jit.predictions_shape()?, 80, 300)?;
for [x1, y1, x2, y2, conf, class] in &detections[0] {
    println!("class {} {conf:.2} at ({x1:.0}, {y1:.0})-({x2:.0}, {y2:.0})", *class as usize);
}
```

---

## Текстовые эмбеддинги

Текстовые модели принимают идентификаторы токенов; токенизация остаётся на
стороне вызывающего кода (пример `qwen3_embed` реализует byte-level BPE Qwen2
на `tiktoken-rs` по опубликованному `tokenizer.json`). `Qwen3Embedder`
упаковывает строки в батчи `max_batch × max_len` и держит по одному
подготовленному плану на каждую форму батча:

```rust
use svod_model::qwen3::{Qwen3Embedder, Qwen3Embedding, qwen3_embedding_0_6b};

let config = qwen3_embedding_0_6b();                  // bf16 on a tensor-core GPU, else f32
let model = Qwen3Embedding::from_hub("Qwen/Qwen3-Embedding-0.6B", config)?;
let mut embedder = Qwen3Embedder::new(model, 8, 512); // batch, max_len

let rows: Vec<Vec<u32>> = texts.iter().map(|t| tokenize(t)).collect();
let embeddings: Vec<Vec<f32>> = embedder.embed(&rows)?; // L2-normalized, one row per text
```

`BgeM3Embedder` (`encode_dense`, `encode_colbert`, `encode` с разреженными
весами), реранкеры BGE и Qwen3 и `ModernBert` следуют тому же соглашению
`from_hub(model_id, config)`; `svod_model::default_compute_dtype()` выбирает
bf16, если у устройства есть tensor cores с поддержкой bf16, и f32 в
противном случае.

---

## Своя модель

Новая модель — это структура с `#[derive(Module)]`, чей `forward` написан на
[тензорном API](./examples); веса загружаются через
`svod_model::state::load_safetensors`, а для инференса модель один раз
оборачивается в `jit_wrapper!`. Страница [JIT-графы](./architecture/jit-graphs)
описывает обёртку — символьные переменные батча, рекуррентное состояние на
устройстве и контракт независимости от данных, — а
[Происхождение ядер](./architecture/kernel-origins) показывает, как читать
отчёт профилировщика в терминах дерева ваших модулей.
