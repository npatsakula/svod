---
sidebar_label: 运行模型
---

# 运行模型

`svod-model` 随附预训练模型的纯 Rust 移植。每个模型在首次使用时从 Hugging Face Hub
下载权重，通过 `jit_wrapper!` 把它的图编译一次，并在 crate 的测试中与上游 PyTorch
实现比对。`svod-arch` 在它们周围补上宿主端的机制：CTC / RN-T 解码器、VAD 分块和
长音频流水线。

| 领域 | 模型 | 模块 | 权重 |
|---|---|---|---|
| 语音识别 | GigaAM v3（CTC、RN-T） | `gigaam` | `vpermilp/GigaAM-v3` |
| 语音识别 | Whisper（tiny … large-v3、turbo） | `whisper` | `vpermilp/whisper` |
| 语音活动检测 | FireRedVAD（批量、流式） | `firered_vad` | `vpermilp/firered_vad` |
| 语音活动检测 | Silero VAD 16k | `silero_vad` | `vpermilp/silero-vad` |
| 语音增强 | GTCRN | `gtcrn` | `vpermilp/gtcrn` |
| 说话人日志 | DiariZen（WavLM + Conformer） | `diarizen` | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` |
| 说话人嵌入 | WeSpeaker ResNet34 | `wespeaker` | `pyannote/wespeaker-voxceleb-resnet34-LM` |
| 文本嵌入 | BGE-M3（dense、sparse、ColBERT）、BGE-reranker-v2-m3 | `bgem3` | `BAAI/bge-m3`、`BAAI/bge-reranker-v2-m3` |
| 文本嵌入 | Qwen3-Embedding-0.6B、Qwen3-Reranker-0.6B | `qwen3` | `Qwen/Qwen3-Embedding-0.6B` |
| 文本嵌入、填空 | ModernBERT base / large | `modernbert` | `answerdotai/ModernBERT-base` |
| 视觉 | ResNet 18 / 34 / 50 / 101 / 152 | `resnet` | `timm/resnet*.a1_in1k` |
| 视觉 | YOLO26（detect、cls、seg、obb、pose、depth、semseg） | `yolo` | `ultralytics/yolo26n` … `yolo26x` |

[model crate 的 README](https://github.com/npatsakula/svod/tree/main/model) 列出了每一行
背后的上游仓库和一致性测试。

---

## 准备

```toml
[dependencies]
svod-model  = "0.1"
svod-arch   = "0.1"   # Asr, splitters, decoders
svod-tensor = "0.1"
svod-dtype  = "0.1"
```

构建需要机器上有 LLVM 和 Clang（见[简介](./introduction#building)）；CPU 不需要其他
任何东西。GPU 后端已编译进来，驱动存在时会自行注册——用 `SVOD_DEVICE` 选择一个：

```bash
SVOD_DEVICE=CUDA:0 cargo run --release -p svod-model --example whisper_infer -- audio.wav
SVOD_DEVICE=AMD:0  cargo run --release -p svod-model --example gigaam_infer -- audio.wav
```

权重存放在 Hugging Face 缓存中（`HF_HUB_CACHE`，默认 `~/.cache/huggingface`；受限
仓库需要 `HF_TOKEN`）。编译好的内核缓存在 `~/.cache/svod/objects`，因此进程第二次
启动时跳过编译。

---

## 示例程序

每个示例都是
[`model/examples`](https://github.com/npatsakula/svod/tree/main/model/examples)
下的完整程序，也是了解它所驱动模型的最佳起点。所有音频示例都要求 16 kHz 单声道 WAV。

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

图像输入是原始的 `f32` NCHW 文件（`3 × side × side × 4` 字节，归一化到 0–1）；省略
`--image` 时示例会合成一个图案，所以上面每条命令无需任何素材就能运行。

语音和嵌入示例上的 `--profile` 打印逐阶段报告，含每个内核的设备时间、roofline
GFLOP/s 和 GB/s。设置 `SVOD_ORIGIN=1` 后，每个内核都会归因到产生它的模型代码，
`--origin-depth N` 把报告汇总到模块深度 `N`：

```bash
SVOD_ORIGIN=1 cargo run -p svod-model --release --example gigaam_infer -- \
    audio.wav --profile --origin-depth 3 --profile-json profile.json
```

---

## 语音识别

### GigaAM

长音频转写是一条流水线：VAD 把波形切成适合编码器大小的语音块，每块分别转写，再带着
时间戳拼接回去。`Asr::assemble` 把两半接起来，使转写器的 JIT 按分块器能产生的最大块
大小只准备一次：

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

头部（CTC 或 RN-T）跟随权重的 revision。`TranscribeOpts` 保存结构性的选择——
`beam_decode` 和 SDPA 分数缓冲区预算 `max_scores_mib`——而 `RunOptions`（`words`、
`segments`、`profile`）是逐调用的开关，所以一个 `Asr` 可服务所有模式。
`FireRedVadSplitter::builder()` 暴露分块参数（`threshold`、`min_duration`、
`max_duration`、`target_duration`……）；`SVOD_VAD_THRESHOLD` 和
`SVOD_VAD_TARGET_CHUNK_SECS` 从环境覆盖默认值。换成 `FixedLengthSplitter` 或
`SileroVadSplitter` 无需改动转写器。

### Whisper

Whisper 解码固定的 30 秒窗口，因此分块器是 `FixedLengthSplitter`。转写器运行
mel → 编码器 → 带温度回退的缓存 beam 解码，按需再做 DTW 词对齐：

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

`WhisperSize::from_name("large-v3")` 解析命令行的拼写，`WhisperPlan`（`encoder_batch`、
`decoder_slots`、`alignment_batch`）在默认值不适合设备时为编译的图定尺寸。

---

## 语音活动检测与增强

`FireRedVadStreamer` 是流式前端：样本到达时喂进去，语音边界以事件形式返回。卷积缓存
在两次推送之间在设备上循环使用，所以每 `chunk_frames` 的音频只花一次小的派发：

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

GTCRN 语音增强是一个覆盖 STFT → 网络 → ISTFT 的 JIT。GRU 递归按帧展开，所以示例以
固定的 128 帧块运行它，只 `prepare` 一次：

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

## 视觉

视觉模型各自是一个 `jit_wrapper!`：一个 `images` 输入、一个 `batch_var`（让一个计划
服务到 `max_batch_size` 为止的任意批大小）和一个输出。图像尺寸在 `prepare` 时固定：

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

这就是它背后的 wrapper，原样照录：

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

YOLO26 形式相同。检测头返回解码后的 `xyxy` 框和类别分数，`[B, 4 + nc, anchors]`，
`postprocess_raw` 做 top-k 选择：

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

## 文本嵌入

文本模型接受 token id；分词由调用方负责（`qwen3_embed` 示例基于 `tiktoken-rs` 用
发布的 `tokenizer.json` 实现了 Qwen2 的字节级 BPE）。`Qwen3Embedder` 把行打包成
`max_batch × max_len` 的批次，每种批次形状保留一个已准备好的计划：

```rust
use svod_model::qwen3::{Qwen3Embedder, Qwen3Embedding, qwen3_embedding_0_6b};

let config = qwen3_embedding_0_6b();                  // bf16 on a tensor-core GPU, else f32
let model = Qwen3Embedding::from_hub("Qwen/Qwen3-Embedding-0.6B", config)?;
let mut embedder = Qwen3Embedder::new(model, 8, 512); // batch, max_len

let rows: Vec<Vec<u32>> = texts.iter().map(|t| tokenize(t)).collect();
let embeddings: Vec<Vec<f32>> = embedder.embed(&rows)?; // L2-normalized, one row per text
```

`BgeM3Embedder`（`encode_dense`、`encode_colbert`、带稀疏权重的 `encode`）、BGE 和
Qwen3 的重排序器以及 `ModernBert` 遵循同样的 `from_hub(model_id, config)` 约定；
`svod_model::default_compute_dtype()` 在设备有 bf16 tensor core 时选择 bf16，否则 f32。

---

## 编写自己的模型

一个新模型就是一个 `#[derive(Module)]` 结构体，其 `forward` 用[张量 API](./examples)
编写，用 `svod_model::state::load_safetensors` 加载，并为推理用 `jit_wrapper!` 包装
一次。[JIT 图](./architecture/jit-graphs)讲解这个 wrapper——符号批次变量、设备上的
循环状态和数据无关性契约——[内核来源](./architecture/kernel-origins)展示如何对照你的
模块树阅读性能分析器的输出。
