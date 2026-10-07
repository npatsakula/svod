---
sidebar_label: मॉडल चलाना
---

# मॉडल चलाना

`svod-model` pretrained मॉडलों के pure-Rust ports देता है। हर मॉडल पहली बार उपयोग पर
अपने weights Hugging Face Hub से डाउनलोड करता है, अपना ग्राफ़
`jit_wrapper!` के ज़रिए एक बार कंपाइल करता है, और क्रेट के टेस्ट में upstream PyTorch implementation
के विरुद्ध जाँचा जाता है। `svod-arch` इनके चारों ओर host-side तंत्र जोड़ता है:
CTC / RN-T decoders, VAD chunking और long-form ऑडियो पाइपलाइन।

| क्षेत्र | मॉडल | मॉड्यूल | Weights |
|---|---|---|---|
| स्पीच रिकग्निशन | GigaAM v3 (CTC, RN-T) | `gigaam` | `vpermilp/GigaAM-v3` |
| स्पीच रिकग्निशन | Whisper (tiny … large-v3, turbo) | `whisper` | `vpermilp/whisper` |
| वॉइस एक्टिविटी | FireRedVAD (batch, streaming) | `firered_vad` | `vpermilp/firered_vad` |
| वॉइस एक्टिविटी | Silero VAD 16k | `silero_vad` | `vpermilp/silero-vad` |
| स्पीच एन्हांसमेंट | GTCRN | `gtcrn` | `vpermilp/gtcrn` |
| स्पीकर डायराइज़ेशन | DiariZen (WavLM + Conformer) | `diarizen` | `BUT-FIT/diarizen-wavlm-large-s80-md-v2` |
| स्पीकर डायराइज़ेशन | Nemotron-3-Diarization (ऑफ़लाइन, स्ट्रीमिंग; अधिकतम 8 स्पीकर) | `nemotron_diar` | `nvidia/Nemotron-3-Diarization` |
| स्पीकर एम्बेडिंग | WeSpeaker ResNet34 | `wespeaker` | `pyannote/wespeaker-voxceleb-resnet34-LM` |
| टेक्स्ट एम्बेडिंग | BGE-M3 (dense, sparse, ColBERT), BGE-reranker-v2-m3 | `bgem3` | `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3` |
| टेक्स्ट एम्बेडिंग | Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B | `qwen3` | `Qwen/Qwen3-Embedding-0.6B` |
| टेक्स्ट एम्बेडिंग, fill-mask | ModernBERT base / large | `modernbert` | `answerdotai/ModernBERT-base` |
| विज़न | ResNet 18 / 34 / 50 / 101 / 152 | `resnet` | `timm/resnet*.a1_in1k` |
| विज़न | YOLO26 (detect, cls, seg, obb, pose, depth, semseg) | `yolo` | `ultralytics/yolo26n` … `yolo26x` |

[मॉडल क्रेट README](https://github.com/npatsakula/svod/tree/main/model)
हर पंक्ति के पीछे के upstream रिपॉज़िटरी और parity टेस्ट सूचीबद्ध करता है।

---

## सेटअप

```toml
[dependencies]
svod-model  = "0.1"
svod-arch   = "0.1"   # Asr, splitters, decoders
svod-tensor = "0.1"
svod-dtype  = "0.1"
```

बिल्ड के लिए मशीन पर LLVM और Clang चाहिए (देखें
[परिचय](./introduction#building)); CPU के लिए और कुछ नहीं चाहिए।
GPU बैकएंड कंपाइल होकर शामिल रहते हैं और ड्राइवर मौजूद होने पर खुद को रजिस्टर करते हैं
— `SVOD_DEVICE` से एक चुनें:

```bash
SVOD_DEVICE=CUDA:0 cargo run --release -p svod-model --example whisper_infer -- audio.wav
SVOD_DEVICE=AMD:0  cargo run --release -p svod-model --example gigaam_infer -- audio.wav
```

Weights Hugging Face कैश में जाते हैं (`HF_HUB_CACHE`, या डिफ़ॉल्ट रूप से `~/.cache/huggingface`;
gated रिपॉज़िटरी के लिए `HF_TOKEN`)। कंपाइल किए गए कर्नेल
`~/.cache/svod/objects` में कैश होते हैं, इसलिए प्रोसेस की दूसरी शुरुआत कंपाइलेशन छोड़ देती है।

---

## उदाहरण

हर उदाहरण
[`model/examples`](https://github.com/npatsakula/svod/tree/main/model/examples)
के अंतर्गत एक पूरा प्रोग्राम है और जिस मॉडल को वह चलाता है उसके लिए सबसे अच्छा शुरुआती बिंदु है। सभी ऑडियो उदाहरण
16 kHz mono WAV की अपेक्षा करते हैं।

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

इमेज इनपुट raw `f32` NCHW फ़ाइलें हैं (`3 × side × side × 4` बाइट, 0–1 पर
normalized); `--image` न देने पर उदाहरण एक पैटर्न संश्लेषित करते हैं, इसलिए ऊपर का हर
कमांड बिना किसी asset के चलता है।

स्पीच और एम्बेडिंग उदाहरणों पर `--profile` हर चरण की एक रिपोर्ट प्रिंट करता है,
जिसमें हर कर्नेल का डिवाइस समय, roofline GFLOP/s और GB/s होता है। `SVOD_ORIGIN=1` के साथ
हर कर्नेल का श्रेय उस मॉडल कोड को दिया जाता है जिसने उसे बनाया, और
`--origin-depth N` रिपोर्ट को मॉड्यूल गहराई `N` तक समेटता है:

```bash
SVOD_ORIGIN=1 cargo run -p svod-model --release --example gigaam_infer -- \
    audio.wav --profile --origin-depth 3 --profile-json profile.json
```

---

## स्पीच रिकग्निशन

### GigaAM

Long-form ट्रांसक्रिप्शन एक पाइपलाइन है: एक VAD waveform को encoder के आकार के
स्पीच chunks में बाँटता है, हर chunk ट्रांसक्राइब होता है, और टुकड़ों को
timestamps के साथ वापस जोड़ा जाता है। `Asr::assemble` दोनों हिस्सों को जोड़ता है ताकि
transcriber का JIT उसी chunk size पर एक बार prepare हो जिसे splitter दे सकता है:

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

Head (CTC या RN-T) weights revision के अनुसार तय होता है। `TranscribeOpts`
संरचनात्मक विकल्प रखता है — `beam_decode` और SDPA scores बजट
`max_scores_mib` — जबकि `RunOptions` (`words`, `segments`, `profile`) हर कॉल का
स्विच है, इसलिए एक `Asr` हर मोड को सर्व करता है। `FireRedVadSplitter::builder()`
chunking के नियंत्रण देता है (`threshold`, `min_duration`, `max_duration`,
`target_duration`, …); `SVOD_VAD_THRESHOLD` और `SVOD_VAD_TARGET_CHUNK_SECS`
environment से डिफ़ॉल्ट बदलते हैं। transcriber को छुए बिना `FixedLengthSplitter` या
`SileroVadSplitter` लगा सकते हैं।

### Whisper

Whisper निश्चित 30-सेकंड windows डिकोड करता है, इसलिए splitter एक
`FixedLengthSplitter` है। transcriber mel → encoder → temperature fallback के साथ cached beam
decoding चलाता है, फिर अनुरोध पर DTW word alignment:

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

`WhisperSize::from_name("large-v3")` CLI वाली वर्तनी पार्स करता है, और
`WhisperPlan` (`encoder_batch`, `decoder_slots`, `alignment_batch`) कंपाइल किए गए ग्राफ़ों
का आकार तय करता है जब डिफ़ॉल्ट डिवाइस में फ़िट नहीं होते।

---

## वॉइस एक्टिविटी और एन्हांसमेंट

`FireRedVadStreamer` streaming front-end है: samples आते ही उन्हें दें,
स्पीच की सीमाएँ events के रूप में वापस पाएँ। conv caches pushes के बीच डिवाइस पर ही
पुनः उपयोग होते हैं, इसलिए ऑडियो के हर `chunk_frames` की लागत एक छोटा dispatch है:

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

GTCRN स्पीच एन्हांसमेंट STFT → नेटवर्क → ISTFT पर एक JIT है। GRU
recurrence हर frame के लिए unroll होता है, इसलिए उदाहरण इसे निश्चित 128-frame chunks में
एक `prepare` के साथ चलाता है:

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

## स्पीकर डायराइज़ेशन

`nemotron_diar` Nemotron-3-Diarization चलाता है: हर 10 ms फ़्रेम के लिए अधिकतम 8
स्पीकरों की गतिविधि, जिन्हें पहली बार आने के क्रम में क्रमांकित किया जाता है।
रिकॉर्डिंग और लाइव स्ट्रीम एक ही लूप से गुज़रते हैं — ऑडियो को चंक में काटा जाता है,
और हर चंक पिछले फ़्रेमों के स्पीकर कैश के साथ एनकोड होता है — फ़र्क़ सिर्फ़ चंक के
आकार का है (ऑफ़लाइन, या मॉडल कार्ड की 1.04 / 0.64 / 0.32 s लेटेंसी):

```rust
use svod_arch::diarization::{Binarization, write_rttm};
use svod_dtype::DType;
use svod_model::nemotron_diar::{Diarizer, NemotronDiar, StreamingMode};

// पूरी रिकॉर्डिंग।
let mut diarizer = Diarizer::offline(NemotronDiar::from_hub(DType::BFloat16, 1)?)?;
let result = diarizer.diarize(&waveform, 16000)?;        // probs: [frames, 8], 10 ms की पंक्तियाँ
write_rttm(std::io::stdout(), "meeting", &result.segments(&Binarization::default()))?;

// लाइव स्ट्रीम, 1.04 s लेटेंसी।
let model = NemotronDiar::from_hub(DType::BFloat16, 1)?;
let mut diarizer = Diarizer::streaming(model, StreamingMode::LowLatency)?;
let mut session = diarizer.session();
for block in microphone {
    session.push(&block)?;
    diarizer.run(&mut [&mut session])?;
    let probs = session.take_probs();                    // पिछली कॉल के बाद निकली पंक्तियाँ
}
session.finish();
diarizer.run(&mut [&mut session])?;
```

`run` कितने भी सेशन लेता है और उनके तैयार चंक को बैच में जोड़ता है, अधिकतम
`from_hub` को दिए गए `max_batch` तक। `svod_arch::diarization` में मॉडल से स्वतंत्र
हिस्से हैं: स्पीकर कैश, प्रायिकताओं को सेगमेंट में बदलने वाली थ्रेशोल्ड, और RTTM आउटपुट।

---

## विज़न

विज़न मॉडल हर एक एक ही `jit_wrapper!` हैं: एक `images` इनपुट, एक
`batch_var` ताकि एक प्लान `max_batch_size` तक हर batch size सर्व करे, और
एक output। इमेज का आकार `prepare` पर तय हो जाता है:

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

इसके पीछे का wrapper, जैसा का तैसा:

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

YOLO26 भी इसी ढाँचे का पालन करता है। detection head class scores के साथ डिकोड किए गए `xyxy`
boxes लौटाता है, `[B, 4 + nc, anchors]`, और `postprocess_raw`
top-k चयन करता है:

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

## टेक्स्ट एम्बेडिंग

टेक्स्ट मॉडल token ids लेते हैं; tokenization कॉल करने वाले के पास रहता है
(`qwen3_embed` उदाहरण प्रकाशित `tokenizer.json` से `tiktoken-rs` पर Qwen2 का byte-level BPE
लागू करता है)। `Qwen3Embedder` पंक्तियों को
`max_batch × max_len` के batches में पैक करता है और हर batch shape के लिए एक prepared प्लान रखता है:

```rust
use svod_model::qwen3::{Qwen3Embedder, Qwen3Embedding, qwen3_embedding_0_6b};

let config = qwen3_embedding_0_6b();                  // bf16 on a tensor-core GPU, else f32
let model = Qwen3Embedding::from_hub("Qwen/Qwen3-Embedding-0.6B", config)?;
let mut embedder = Qwen3Embedder::new(model, 8, 512); // batch, max_len

let rows: Vec<Vec<u32>> = texts.iter().map(|t| tokenize(t)).collect();
let embeddings: Vec<Vec<f32>> = embedder.embed(&rows)?; // L2-normalized, one row per text
```

`BgeM3Embedder` (`encode_dense`, `encode_colbert`, sparse
weights के साथ `encode`), BGE और Qwen3 rerankers और `ModernBert` उसी
`from_hub(model_id, config)` परिपाटी का पालन करते हैं; `svod_model::default_compute_dtype()`
जहाँ डिवाइस में bf16 tensor cores हों वहाँ bf16 और अन्यथा f32 चुनता है।

---

## अपना मॉडल लिखना

नया मॉडल एक `#[derive(Module)]` struct है जिसका `forward`
[टेंसर API](./examples) पर लिखा जाता है, `svod_model::state::load_safetensors` से लोड होता है,
और इन्फ़रेंस के लिए एक बार `jit_wrapper!` में लपेटा जाता है। [JIT ग्राफ़](./architecture/jit-graphs)
wrapper को कवर करता है — symbolic batch variables, डिवाइस पर recurrent state और
data-independence contract — और [कर्नेल origins](./architecture/kernel-origins)
दिखाता है कि अपने मॉड्यूल ट्री के सापेक्ष प्रोफ़ाइलर को कैसे पढ़ें।
