---
sidebar_label: परिचय
---

<div align="center">

# Svod

**Rust में लिखा गया डीप लर्निंग कंपाइलर और इन्फ़रेंस इंजन।**

[![CI](https://github.com/npatsakula/svod/actions/workflows/ci.yml/badge.svg)](https://github.com/npatsakula/svod/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-svod.vpermilp.online-blue)](https://svod.vpermilp.online/docs/introduction)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/npatsakula/svod/tree/main/LICENSE)

[दस्तावेज़](https://svod.vpermilp.online/docs/introduction) ·
[मॉडल](https://github.com/npatsakula/svod/tree/main/#models-and-pipelines) ·
[आर्किटेक्चर](https://svod.vpermilp.online/docs/architecture/pipeline) ·
[व्याख्यान](https://github.com/npatsakula/svod/tree/main/#talks-and-writing) ·
[रोडमैप](https://github.com/npatsakula/svod/tree/main/#roadmap)

</div>

Svod लेज़ी टेंसर ग्राफ़ को CPU, AMD और NVIDIA GPU के लिए फ़्यूज़्ड कर्नेल में कंपाइल
करता है, और बीच में कोई वेंडर रनटाइम नहीं होता: न PyTorch, न ROCm/HIP, न CUDA toolkit।
यह [Tinygrad](https://github.com/tinygrad/tinygrad) के डिज़ाइन पर चलता है: एक छोटा,
सत्यापन-योग्य IR (UOps), पैटर्न-आधारित रीराइट, और टेंसर से मशीन कोड तक एक सीधी
पाइपलाइन।

Svod के साथ स्पीच, टेक्स्ट और विज़न मॉडल आते हैं, जिन्हें उनके PyTorch संदर्भों के
विरुद्ध जाँचा गया है; बाकी सब के लिए एक ONNX इम्पोर्टर है।

## Svod क्यों

- **शुरू से अंत तक एक ही प्रतिनिधित्व।** PyTorch एक मॉडल को सात या उससे अधिक IR से
  गुज़ारता है, और हर सीमा एक पुल है जिसे वेंडर को बनाना पड़ता है और डिबगिंग संदर्भ में एक
  दरार है। Svod में एक ही UOp ग्राफ़ बनता है, ऑप्टिमाइज़ होता है, शेड्यूल होता है और
  रेंडर होता है। नए एक्सेलेरेटर के लिए सिर्फ़ तीन चीज़ें चाहिए: एक कोड जनरेटर, एक बफ़र
  एलोकेटर और एक कर्नेल लॉन्चर।
- **प्रोडक्शन Rust।** GIL के इर्द-गिर्द Python ग्लू की जगह स्टैटिक टाइप और नेटिव
  कॉन्करेंसी। डिप्लॉयमेंट एक बाइनरी है, बिना LibTorch या ONNX Runtime बाइंडिंग और बिना
  वेंडर SDK के।
- **जाना-पहचाना API।** टेंसर API PyTorch को नामित आर्ग्युमेंट तक प्रतिबिंबित करता है,
  इसलिए मॉडल पोर्ट संदर्भ की तरह पढ़ा जाता है और पोर्टिंग अधिकतर यांत्रिक होती है।
- **जहाँ ज़रूरी है वहाँ गति।** जब कंपाइलर पर्याप्त न हो, `tk` टाइल DSL उसी IR में
  हाथ से लिखे कर्नेल देता है। वे प्रोफ़ाइलर और ऑरिजिन ट्रैकर को दिखते रहते हैं, अपारदर्शी
  बाइनरी की तरह अलग-थलग नहीं रहते।

प्रेरणा और डिज़ाइन
[Developing Svod](https://blog.vpermilp.online/en/blog/svod-intro/) में बताए गए हैं।

## मॉडल और पाइपलाइन

| क्षेत्र | मॉडल |
|---|---|
| स्पीच रिकग्निशन | Whisper, GigaAM v3 (CTC, RN-T) |
| वॉइस एक्टिविटी | FireRedVAD, Silero VAD |
| स्पीच एन्हांसमेंट | GTCRN |
| स्पीकर विश्लेषण | DiariZen, WeSpeaker |
| टेक्स्ट एम्बेडिंग और रीरैंकिंग | BGE-M3, Qwen3-Embedding, ModernBERT |
| विज़न | YOLO26, ResNet |
| बाकी सब | [ONNX इम्पोर्टर](https://github.com/npatsakula/svod/tree/main/onnx/) ([ऑपरेटर कवरेज](https://github.com/npatsakula/svod/tree/main/onnx/PARITY.md)) |

वेट सीधे Hugging Face Hub से आते हैं, और आउटपुट संदर्भ कार्यान्वयनों के विरुद्ध जाँचे
जाते हैं। [`model/`](https://github.com/npatsakula/svod/tree/main/model/) में वेरिएंट, अपस्ट्रीम लिंक और चलाने योग्य
उदाहरण सूचीबद्ध हैं; [`arch`](https://github.com/npatsakula/svod/tree/main/arch/) में डिकोडर और लंबे ऑडियो की पाइपलाइन है।

## इंजन

### ग्राफ़ कैप्चर: एक बार कंपाइल, कई बार रीप्ले

मॉडल को एक बार ट्रेस करके एक्ज़ीक्यूशन प्लान बनाया जाता है, और उसके बाद हर कॉल सिर्फ़
उसे रीप्ले करती है। सिम्बॉलिक डायमेंशन (बैच, सीक्वेंस लंबाई) हर कॉल पर बिना पुनः
कंपाइल किए बाइंड होते हैं। रिकरेंट स्टेट कॉल्स के बीच डिवाइस पर ही रहती है, और एक
मेमोरी प्लानर TLSF एरीना के ज़रिए मध्यवर्ती बफ़र पुनः उपयोग करता है। स्टैटिक चेन एक
हार्डवेयर ग्राफ़ के रूप में रीप्ले होती हैं: NVIDIA पर **CUDA Graphs**, AMD पर
प्रति-रीप्ले एक डोरबेल वाला AQL/PM4 ग्राफ़ (HCQGraph की तरह), और Metal पर indirect
command buffers। देखें [JIT ग्राफ़](https://svod.vpermilp.online/docs/architecture/jit-graphs)।

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

### Z3 से सिद्ध रीराइट

हर ऑप्टिमाइज़ेशन `patterns!` DSL में एक घोषणात्मक रीराइट है। बीजगणितीय और इंडेक्स
सरलीकरणों की जाँच **Z3 SMT सॉल्वर** से होती है: यह सिद्ध करता है कि पुनर्लिखित
अभिव्यक्ति हर इनपुट के लिए मूल के बराबर है, या एक प्रति-उदाहरण लौटाता है। पाइपलाइन का
बाकी हिस्सा प्रॉपर्टी-आधारित टेस्ट कवर करते हैं। देखें
[पैटर्न सिस्टम](https://svod.vpermilp.online/docs/architecture/optimizations/pattern-system)।

### प्लेटफ़ॉर्म-विशिष्ट कोड जनरेशन

- **Tensor cores** आर्किटेक्चर के अनुसार चुने जाते हैं: NVIDIA sm_75/80/89, AMD RDNA3,
  RDNA4 और CDNA3/4, साथ ही Apple Metal। fp8 sm_89 और CDNA3 पर उपलब्ध है।
- **टाइल कर्नेल (`tk`)**: GEMM, flash attention, RMSNorm और k-means के लिए Rust में
  ThunderKittens-शैली का टाइल DSL। एक ही कर्नेल सोर्स AMD MFMA/WMMA (gfx942, gfx11,
  gfx12), CUDA `mma.sync` (sm_80+) और Apple
  `simdgroup_matrix` (Apple7+) में लोअर होता है। टाइल आकार पहले उपयोग पर
  ऑटोट्यून होकर कैश हो जाते हैं। देखें
  [टाइल कर्नेल](https://svod.vpermilp.online/docs/tile-kernels/overview)।

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
- **कर्नेल खोज**: हाथ से लिखे ह्यूरिस्टिक, या ऑप्टिमाइज़ेशन स्पेस पर BEAM खोज, जिसका
  कैश डिस्क पर स्थायी रहता है। देखें
  [कर्नेल खोज](https://svod.vpermilp.online/docs/architecture/optimizations/kernel-search)।
- **CPU**: प्रोसेस के भीतर कंपाइल होने वाला वेक्टराइज़्ड LLVM IR, x86_64, aarch64,
  riscv64, loongarch64 और ppc64le के लिए अपना ELF लोडर, और मल्टी-थ्रेडेड कर्नेल।

### ज़ीरो-कॉपी डेटा पथ

ONNX इनिशियलाइज़र और `Tensor::from_path` टेंसर डिस्क से लेज़ी तरीके से मेमोरी-मैप होते हैं। डिवाइस बफ़र
सब-व्यू समर्थित करते हैं। होस्ट कोड रियलाइज़्ड टेंसरों को उधार लिए गए
`ndarray` व्यू (`array_view`, `array_view_mut`) से पढ़ता-लिखता है, इसलिए कैप्चर किए गए प्लान
को डेटा देने में कुछ भी कॉपी नहीं होता।

### कर्नेल फ़्यूज़न और एट्रिब्यूशन

RANGEIFY शेड्यूलर एलिमेंटवाइज़, रिडक्शन और मूवमेंट ऑप्स को यथासंभव कम कर्नेलों में
फ़्यूज़ करता है। हर कर्नेल दर्ज रखता है कि वह कहाँ से आया (मॉड्यूल पथ, ONNX नोड या
सोर्स लाइन), इसलिए प्रोफ़ाइलर डिवाइस समय, roofline GFLOP/s और GB/s, ऑक्यूपेंसी और
हार्डवेयर काउंटर (AMD SQ, NVIDIA CUPTI) को वापस मॉडल कोड से जोड़ सकता है। देखें
[कर्नेल ऑरिजिन](https://svod.vpermilp.online/docs/architecture/kernel-origins)।

## बैकएंड

| डिवाइस | सिलेक्टर | कंपाइलेशन | रनटाइम |
|---|---|---|---|
| CPU | `CPU` (macOS के बाहर डिफ़ॉल्ट) | रनटाइम पर लोड होने वाली `libLLVM` से LLVM IR, न मिलने पर `clang`; Clang C बैकएंड | अपना ELF JIT लोडर, मल्टी-थ्रेडेड |
| AMD GPU | `AMD:N` | `clang --target=amdgcn-amd-amdhsa` | सीधी KFD क्यू (AQL/PM4), बिना HIP या ROCm रनटाइम |
| NVIDIA GPU | `CUDA:N` | `clang` NVPTX → PTX → `ptxas` या ड्राइवर JIT | रनटाइम पर लोड `libcuda.so.1`, बिना CUDA toolkit |
| Apple GPU | `METAL:N` (macOS पर डिफ़ॉल्ट) | MSL → metallib | रनटाइम पर लोड Metal फ़्रेमवर्क |

हर GPU बैकएंड कंपाइल में शामिल है और केवल हार्डवेयर मौजूद होने पर पंजीकृत होता है।
किसी एक को चुनने के लिए `SVOD_DEVICE` सेट करें या `Tensor::to(device)` कॉल करें।

CPU कोड x86_64, aarch64, riscv64 और ppc64le पर, Linux और macOS पर टेस्ट होता है। GPU:
AMD RDNA 3.5, RDNA 4 और CDNA 3, NVIDIA sm_80 व उससे नए, और Apple M3 व उससे नए।

## व्याख्यान और लेख

[Developing Svod](https://blog.vpermilp.online/en/blog/svod-intro/), एक ब्लॉग पोस्ट
(अगस्त 2026) कि Svod क्यों है, उसका आर्किटेक्चर और रोडमैप।

| आयोजन | व्याख्यान | भाषा |
|---|---|---|
| [Data Fest 2026](https://ods.ai/events/df2026-31-may-online) (ऑनलाइन, 31 मई 2026) | Svod पर सबसे तेज़ Sber GigaAM इन्फ़रेंस लिखना | रूसी |
| [RustCon 2025](https://rustcon.ru/morok-minimalistichnyy-deep-learning-freymvork-na-rust) (मॉस्को, नवंबर 2025) | Morok: Rust में एक मिनिमलिस्ट डीप लर्निंग फ़्रेमवर्क | रूसी |
| [Stereo Data Ёлка 2025](https://ods.ai/events/data-elka-2025-vk-offline-spb) (सेंट पीटर्सबर्ग, जनवरी 2026) | Rust में ML | रूसी |

Morok Svod का पुराना नाम था।

## वर्कस्पेस

| क्रेट | भूमिका |
|---|---|
| [`dtype`](https://github.com/npatsakula/svod/tree/main/dtype/) | स्केलर, वेक्टर, पॉइंटर और इमेज टाइप, जिनमें bf16 और fp8 शामिल हैं |
| [`ir`](https://github.com/npatsakula/svod/tree/main/ir/) | हैश-कॉन्सिंग, सिम्बॉलिक पूर्णांक और ऑरिजिन वाला UOp ग्राफ़ IR |
| [`macros`](https://github.com/npatsakula/svod/tree/main/macros/) | `patterns!` रीराइट DSL और `jit_wrapper!` |
| [`schedule`](https://github.com/npatsakula/svod/tree/main/schedule/) | RANGEIFY, रीराइट पास, ह्यूरिस्टिक और BEAM, Z3 सत्यापन |
| [`codegen`](https://github.com/npatsakula/svod/tree/main/codegen/) | LLVM IR (CPU, AMDGPU, NVPTX), C और MSL रेंडरर |
| [`device`](https://github.com/npatsakula/svod/tree/main/device/) | बफ़र, एलोकेटर, mmap, KFD, CUDA और Metal ड्राइवर, हार्डवेयर ग्राफ़ |
| [`runtime`](https://github.com/npatsakula/svod/tree/main/runtime/) | कर्नेल कंपाइलेशन, कैशिंग, एक्ज़ीक्यूशन प्लान और प्रोफ़ाइलर |
| [`tensor`](https://github.com/npatsakula/svod/tree/main/tensor/) | लेज़ी टेंसर API, `nn` मॉड्यूल और मेमोरी प्लानर |
| [`tk`](https://github.com/npatsakula/svod/tree/main/tk/) | टाइल कर्नेल DSL और कर्नेल लाइब्रेरी |
| [`onnx`](https://github.com/npatsakula/svod/tree/main/onnx/) | ONNX इम्पोर्टर |
| [`arch`](https://github.com/npatsakula/svod/tree/main/arch/) | होस्ट-साइड डिकोडर, VAD सेगमेंटेशन और ऑडियो पाइपलाइन |
| [`model`](https://github.com/npatsakula/svod/tree/main/model/) | प्रीट्रेन्ड मॉडल और उदाहरण |

## लाइब्रेरी का उपयोग

मॉडल पाइपलाइनों में जुड़ते हैं। यहाँ GigaAM से लंबी रूसी स्पीच रिकग्निशन है, जिसे
FireRedVAD सेगमेंट करता है:

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

किसी भी ONNX मॉडल को एक बार कंपाइल करके रीप्ले किया जा सकता है। ग्राफ़ आपके इनपुट टेंसर पर
ट्रेस होता है, इसलिए उसमें लिखा गया नया डेटा हर रीप्ले में दिखता है:

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

## बिल्ड {#building}

Nix flake हर कंपाइलर और लाइब्रेरी को पिन करता है, और CI वही flake उपयोग करता है:

```bash
nix develop      # development shell
nix flake check  # the CI suite: clippy, nextest (with Z3 and proptest), fmt
```

Nix के बिना आपको निम्नलिखित चाहिए:

| निर्भरता | संस्करण | आवश्यक | उद्देश्य |
|---|---|---|---|
| Rust | 1.88+ | हाँ | Edition 2024 |
| LLVM | ≥ 16 | हाँ | CPU कोड जनरेशन; `libLLVM` रनटाइम पर लोड होती है |
| Clang | — | हाँ | GPU कर्नेल कंपाइलेशन, C बैकएंड, `libLLVM` न होने पर फ़ॉलबैक |
| protobuf, pkgconf, zlib, libffi, libxml2 | — | हाँ | ONNX प्रोटो और LLVM टूलचेन |
| Z3 | ≥ 4.15 | नहीं | रीराइट सत्यापन (`--features z3`) |
| NVIDIA ड्राइवर | CUDA ≥ 12.0 (R525) | नहीं | CUDA बैकएंड |
| amdgpu कर्नेल ड्राइवर (KFD) | — | नहीं | AMD बैकएंड |

```bash
cargo test --workspace
cargo test --workspace --features z3,proptest
```

`SVOD_THREADS` कर्नेल कंपाइल करने और CPU कर्नेल चलाने के लिए उपयोग होने वाला एकल
थ्रेड बजट सेट करता है।

## रोडमैप

- **AOT कंपाइलेशन:** ऑप्टिमाइज़्ड ग्राफ़ और कंपाइल किए गए कर्नेल को सीरियलाइज़ करना,
  ताकि मॉडल तुरंत शुरू हो और वहाँ भी चले जहाँ कोई कंपाइलर उपलब्ध नहीं (जैसे WASM)।
- **डेटा विश्लेषण प्रिमिटिव:** हर बैकएंड पर k-means, kNN, PCA, SVD, (H)DBSCAN, UMAP और
  t-SNE के लिए FlashAttention-शैली के GPU कर्नेल। k-means और kNN पहले से `tk` में हैं।
- **जनरेट किए गए कोड का औपचारिक सत्यापन:** एनोटेटेड C आउटपुट जो आउट-ऑफ़-बाउंड्स
  एक्सेस और लॉसी कास्ट की अनुपस्थिति सिद्ध करता है।
- **और हार्डवेयर:** सर्वर (MI300–MI450, H100–B200), कंज़्यूमर (Ryzen AI, Apple M3–M5,
  RTX 30–50) और एम्बेडेड (Snapdragon X, RK3588) लक्ष्य एक ही टेंसर API के पीछे, साथ ही
  AMD सॉफ़्टवेयर स्टैक पर निर्भरता-रहित एक यूज़रस्पेस AMD ड्राइवर।

## लाइसेंस

[MIT](https://github.com/npatsakula/svod/tree/main/LICENSE)
