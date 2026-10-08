---
sidebar_label: Running Models
---

# Running Models

`svod-model` ships pure-Rust ports of pretrained models. Each one downloads its
weights from the Hugging Face Hub on first use, compiles its graph once through
`jit_wrapper!`, and is checked against the upstream PyTorch implementation in
the crate's tests. `svod-arch` adds the host-side machinery around them:
CTC / RN-T decoders, VAD chunking and the long-form audio pipeline.

| Domain | Model | Module | Weights |
|---|---|---|---|
| Speech recognition | GigaAM v3 (CTC, RN-T) | `gigaam` | `vpermilp/GigaAM-v3` |
| Speech recognition | Whisper (tiny … large-v3, turbo) | `whisper` | `vpermilp/whisper` |
| Voice activity | FireRedVAD (batch, streaming) | `firered_vad` | `vpermilp/firered_vad` |
| Voice activity | Silero VAD 16k | `silero_vad` | `vpermilp/silero-vad` |
| Speech enhancement | GTCRN | `gtcrn` | `vpermilp/gtcrn` |
| Speaker diarization | DiariZen (WavLM + Conformer) | `diarizen` | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` |
| Speaker diarization | Nemotron-3-Diarization (offline, streaming; up to 8 speakers) | `nemotron_diar` | `nvidia/Nemotron-3-Diarization` |
| Speaker embedding | WeSpeaker ResNet34 | `wespeaker` | `pyannote/wespeaker-voxceleb-resnet34-LM` |
| Text embeddings | BGE-M3 (dense, sparse, ColBERT), BGE-reranker-v2-m3 | `bgem3` | `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3` |
| Text embeddings | Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B | `qwen3` | `Qwen/Qwen3-Embedding-0.6B` |
| Text embeddings, fill-mask | ModernBERT base / large | `modernbert` | `answerdotai/ModernBERT-base` |
| Vision | ResNet 18 / 34 / 50 / 101 / 152 | `resnet` | `timm/resnet*.a1_in1k` |
| Vision | YOLO26 (detect, cls, seg, obb, pose, depth, semseg) | `yolo` | `ultralytics/yolo26n` … `yolo26x` |

The [model crate README](https://github.com/npatsakula/svod/tree/main/model)
lists the upstream repositories and the parity tests behind each row.

---

## Setup

```toml
[dependencies]
svod-model  = "0.2"
svod-arch   = "0.2"   # Asr, splitters, decoders
svod-tensor = "0.2"
svod-dtype  = "0.2"
```

The build needs LLVM and Clang on the machine (see the
[introduction](./introduction#building)); nothing else is required for the CPU.
GPU backends are compiled in and register themselves when the driver is
present — pick one with `SVOD_DEVICE`:

```bash
SVOD_DEVICE=CUDA:0 cargo run --release -p svod-model --example whisper_infer -- audio.wav
SVOD_DEVICE=AMD:0  cargo run --release -p svod-model --example gigaam_infer -- audio.wav
```

Weights land in the Hugging Face cache (`HF_HUB_CACHE`, or `~/.cache/huggingface`
by default; `HF_TOKEN` for gated repositories). Compiled kernels are cached in
`~/.cache/svod/objects`, so the second start of a process skips compilation.

---

## The examples

Every example is a complete program under
[`model/examples`](https://github.com/npatsakula/svod/tree/main/model/examples)
and is the best starting point for the model it drives. All audio examples
expect 16 kHz mono WAV.

```bash
cargo run -p svod-model --release --example gigaam_infer   -- audio.wav [--rnnt] [--timestamps]
cargo run -p svod-model --release --example whisper_infer  -- audio.wav [--size base] [--language auto] [--timestamps]
cargo run -p svod-model --release --example vad_stream     -- audio.wav                # streaming VAD events
cargo run -p svod-model --release --example vad_bench      -- audio.wav                # Silero vs FireRedVAD
cargo run -p svod-model --release --example gtcrn_enhance  -- --in noisy.wav --out clean.wav --hub
cargo run -p svod-model --release --example nemotron_diarize -- audio.wav [--stream low|very-low|ultra-low]
cargo run -p svod-model --release --example qwen3_embed    -- --texts texts.txt --batch 8 --max-len 512
cargo run -p svod-model --release --example resnet_classify -- --hub --image dog.bin --side 224
cargo run -p svod-model --release --example yolo_detect    -- --hub --scale small --image photo.bin --side 640
```

Image inputs are raw `f32` NCHW files (`3 × side × side × 4` bytes, normalized
to 0–1); the examples synthesize a pattern when `--image` is omitted, so every
command above runs without assets.

`--profile` on the speech and embedding examples prints a per-stage report
with device time, roofline GFLOP/s and GB/s per kernel. With `SVOD_ORIGIN=1`
each kernel is attributed to the model code that produced it, and
`--origin-depth N` rolls the report up to module depth `N`:

```bash
SVOD_ORIGIN=1 cargo run -p svod-model --release --example gigaam_infer -- \
    audio.wav --profile --origin-depth 3 --profile-json profile.json
```

---

## Speech recognition

### GigaAM

Long-form transcription is a pipeline: a VAD splits the waveform into
speech chunks sized to the encoder, each chunk is transcribed, and the pieces
are stitched back with timestamps. `Asr::assemble` wires the two halves so the
transcriber's JIT is prepared once at the chunk size the splitter can emit:

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

The head (CTC or RN-T) follows the weights revision. `TranscribeOpts` holds
the structural choices — `beam_decode` and the SDPA scores budget
`max_scores_mib` — while `RunOptions` (`words`, `segments`, `profile`) is a
per-call switch, so one `Asr` serves every mode. `FireRedVadSplitter::builder()`
exposes the chunking knobs (`threshold`, `min_duration`, `max_duration`,
`target_duration`, …); `SVOD_VAD_THRESHOLD` and `SVOD_VAD_TARGET_CHUNK_SECS`
override the defaults from the environment. Swap in `FixedLengthSplitter` or
`SileroVadSplitter` without touching the transcriber.

### Whisper

Whisper decodes fixed 30-second windows, so the splitter is a
`FixedLengthSplitter`. The transcriber runs mel → encoder → cached beam
decoding with temperature fallback, then DTW word alignment on request:

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

`WhisperSize::from_name("large-v3")` parses the CLI spelling, and
`WhisperPlan` (`encoder_batch`, `decoder_slots`, `alignment_batch`) sizes the
compiled graphs when the defaults do not fit the device.

---

## Voice activity and enhancement

`FireRedVadStreamer` is the streaming front-end: feed samples as they arrive,
get speech boundaries back as events. The conv caches recycle on the device
between pushes, so each `chunk_frames` of audio costs one small dispatch:

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

GTCRN speech enhancement is one JIT over STFT → network → ISTFT. The GRU
recurrence unrolls per frame, so the example runs it in fixed 128-frame chunks
with one `prepare`:

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

## Speaker diarization

`nemotron_diar` runs Nemotron-3-Diarization: per 10 ms frame, the activity of
up to 8 speakers, numbered by first appearance. Recordings and live streams run
the same loop — the audio is cut into chunks, each encoded together with a
speaker cache of earlier frames — and differ only in chunk sizes (offline, or
the model card's 1.04 / 0.64 / 0.32 s latencies):

```rust
use svod_arch::diarization::{Binarization, write_rttm};
use svod_dtype::DType;
use svod_model::nemotron_diar::{Diarizer, NemotronDiar, StreamingMode};

// Whole recording.
let mut diarizer = Diarizer::offline(NemotronDiar::from_hub(DType::BFloat16, 1)?)?;
let result = diarizer.diarize(&waveform, 16000)?;        // probs: [frames, 8], 10 ms rows
write_rttm(std::io::stdout(), "meeting", &result.segments(&Binarization::default()))?;

// Live stream, 1.04 s latency.
let model = NemotronDiar::from_hub(DType::BFloat16, 1)?;
let mut diarizer = Diarizer::streaming(model, StreamingMode::LowLatency)?;
let mut session = diarizer.session();
for block in microphone {
    session.push(&block)?;
    diarizer.run(&mut [&mut session])?;
    let probs = session.take_probs();                    // rows emitted since the last call
}
session.finish();
diarizer.run(&mut [&mut session])?;
```

`run` takes any number of sessions and batches their ready chunks, up to the
`max_batch` passed to `from_hub`. `svod_arch::diarization` holds the
model-agnostic parts: the speaker cache, the thresholds turning probabilities
into segments, and RTTM output.

---

## Vision

The vision models are a single `jit_wrapper!` each: one `images` input, a
`batch_var` so one plan serves every batch size up to `max_batch_size`, and
one output. Image size is baked in at `prepare`:

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

This is the wrapper behind it, verbatim:

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

YOLO26 follows the same shape. The detection head returns decoded `xyxy`
boxes with class scores, `[B, 4 + nc, anchors]`, and `postprocess_raw` does
the top-k selection:

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

## Text embeddings

The text models take token ids; tokenization stays with the caller (the
`qwen3_embed` example implements Qwen2's byte-level BPE on `tiktoken-rs` from
the published `tokenizer.json`). `Qwen3Embedder` packs rows into batches of
`max_batch × max_len` and keeps one prepared plan per batch shape:

```rust
use svod_model::qwen3::{Qwen3Embedder, Qwen3Embedding, qwen3_embedding_0_6b};

let config = qwen3_embedding_0_6b();                  // bf16 on a tensor-core GPU, else f32
let model = Qwen3Embedding::from_hub("Qwen/Qwen3-Embedding-0.6B", config)?;
let mut embedder = Qwen3Embedder::new(model, 8, 512); // batch, max_len

let rows: Vec<Vec<u32>> = texts.iter().map(|t| tokenize(t)).collect();
let embeddings: Vec<Vec<f32>> = embedder.embed(&rows)?; // L2-normalized, one row per text
```

`BgeM3Embedder` (`encode_dense`, `encode_colbert`, `encode` with sparse
weights), the BGE and Qwen3 rerankers and `ModernBert` follow the same
`from_hub(model_id, config)` convention; `svod_model::default_compute_dtype()`
picks bf16 where the device has bf16 tensor cores and f32 otherwise.

---

## Writing your own

A new model is a `#[derive(Module)]` struct whose `forward` is written against
the [tensor API](./examples), loaded with `svod_model::state::load_safetensors`,
and wrapped once in `jit_wrapper!` for inference. [JIT graphs](./architecture/jit-graphs)
covers the wrapper — symbolic batch variables, on-device recurrent state and
the data-independence contract — and [Kernel origins](./architecture/kernel-origins)
shows how to read the profiler against your module tree.
