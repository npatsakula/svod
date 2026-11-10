---
sidebar_label: अवलोकन
---

# Tile कर्नेल (tk) {#tile-kernels-tk}

Svod का ऑप्टिमाइज़र किसी मॉडल के ज़्यादातर हिस्से के लिए तेज़ schedules loop transformations पर सर्च
करके खोज लेता है। कुछ कर्नेल इस तरह नहीं मिलते। Flash attention एक recurrence है: keys का हर block
एक running maximum और sum को अपडेट करता है और accumulator को rescale करता है, इसलिए tile करने के लिए
कोई एक reduction है ही नहीं। Tensor-core GPU पर तेज़ GEMM एक multi-stage `cp.async` ring,
`ldmatrix` fragment loads और swizzled shared layout पर टिका होता है, और ये loop search के कदम नहीं
हैं। ऐसे कर्नेल हाथ से लिखे जाते हैं, और `svod-tk` ("tk") उन्हें लिखने का crate है।

`svod-tk` tile language का तीसरा संस्करण ("tk3") लागू करता है और एकमात्र tile-kernel crate है। आठ मॉडल op layer पर चलते हैं: Nemotron-3-Diarization,
Whisper (encoder, और cache mode में decoder step का attention), ModernBERT,
BGE-M3/XLM-RoBERTa, Qwen3-Embedding, GigaAM, WavLM और YOLO26 (channels-last, जिसके 3×3
convolutions `ops::conv2d` पर चलते हैं)।

## tk क्या है {#what-tk-is}

| हिस्सा | मॉड्यूल | क्या करता है |
|---|---|---|
| Tile program | `ir`, `build` | tile values पर एक structured statement tree (`Let`, `Copy`, `Loop`, `Pipeline`, `If`, `Warps`), जिसे एक `Kernel` builder program order में रिकॉर्ड करता है। यह DAG नहीं है। |
| Layouts | `layout`, `layouts` | हर fragment और swizzle के लिए F2 linear layouts (bit matrices), जिन्हें inference तय करता है। Kernel code कभी किसी lane का नाम नहीं लेता। |
| Atoms | `atoms` | हर target के matrix-core instructions अपने operand layouts के साथ: `mma.sync` (CUDA), `wgmma` (sm_90), WMMA (RDNA3/3.5/4), MFMA 16×16×16, और attention के लिए 32×32×8 (CDNA)। |
| Schedule templates | `schedule` | एक `Pipeline` को `cp.async` groups, एक या दो slots वाली register-staged ring, या ping-pong में दो crews (CDNA GEMM) में फैलाते हैं। लेखक सिर्फ़ produce और consume bodies लिखता है। |
| Lowering | `lower` | Effects से barriers डालता है, schedule को फैलाता है, layouts infer करता है और एक pre-linearized instruction list (`Op::Linear`) emit करता है, ताकि क्रम कोई toposort तय न करे। |
| Kernels | `kernels` | Epilogues वाला GEMM (अधूरी आख़िरी wave K के साथ split होती है), split-K के साथ implicit-GEMM convolution, flash attention, attention prologue, LayerNorm/RMSNorm, और nearest-neighbour search (argmin और top-k)। हर एक में spec struct जाता है और `Program` निकलता है; जिस kernel का config planner चुनता है, वह उसी config के तहत admitted spec (`Admitted`) लेता है। |
| Launch | `launch` | किसी program को lazy ग्राफ़ में custom kernel की तरह चलाता है, lowered bodies को memoize करते हुए। |
| Op layer | `ops` | `linear`, `conv2d`, `attention`, `heads`, `layer_norm`, `rms_norm`, `nearest`, `k_nearest`, … ये हमेशा एक `Tensor` लौटाते हैं: kernel फ़िट हो तो kernel, वरना ग्राफ़। |
| Planner | `ops::config` | हर shape के configs को device की मापी गई costs से रैंक करता है और वह config admit करता है जिसके तहत कोई spec चलता है। |
| Interpreter | `interp` | किसी भी tile program को host पर चलाकर बिना GPU के reference numerics देता है। |
| Tune store | `tune` | हर op के candidates को हर shape और device पर एक बार मापता है और विजेता को डिस्क पर रखता है। |

```text
model code ──► svod_tk::ops          always a Tensor; picks a kernel and config or builds the graph op
                   │
                   ▼
              kernels::*               spec ─► tile Program (builder, program order)
                   │
                   ▼
              lower::lower             operand loads ─► barriers ─► schedule template
                   │                   ─► layout inference ─► emission
                   ▼
              Op::Linear program  ──►  custom kernel in the lazy graph ─► LLVM ─► device
```

## Targets {#targets}

| Target | Matrix core | Fills |
|---|---|---|
| CUDA sm_80+ | `mma.sync` m16n8k16, `ldmatrix` | `cp.async` |
| CUDA sm_90 | `wgmma` जोड़ता है (GEMM और attention) | `cp.async` |
| AMD RDNA4 | WMMA 16×16×16 | registers, 1–2 slots |
| AMD RDNA3 / 3.5 | WMMA 16×16×16 | registers, 1–2 slots |
| AMD CDNA3 / 4 | MFMA 16×16×16, और attention के लिए 32×32×8; wave64 | registers, ping-pong में दो crews |
| Apple | simdgroup 8×8×8 atom | — (कोई kernels नहीं) |

कोई target तभी मापा हुआ गिना जाता है जब उस पर
`targets::families_match_the_interpreter_on_the_device` pass हो जाए। कोई कार्ड कितना पहुँचता है,
यह देखने के लिए उस पर `gemm_throughput_probe` और `attention_throughput_probe` चलाएँ
([Tuning](./tuning#probe-tests)): ये tuner के चुनाव को TFLOP/s में print करते हैं।

:::note[इस रिलीज़ में नहीं]
Hopper TMA और mbarrier producers (sm_90 `cp.async` से भरे जाने वाले `wgmma` चलाता है), Blackwell
`tcgen05`, GEMM के लिए 32×32 MFMA, Apple kernels, fp8, nearest-neighbour search के अलावा int8, attention
backward, और convolution का image-staged रूप। देखें [पोर्टेबिलिटी](./portability#not-in-this-release)।
:::

## पढ़ने का क्रम {#reading-order}

1. [आपका पहला कर्नेल](./first-kernel): builder, GEMM के सहारे।
2. [Layouts और Lowering](./layouts-and-lowering): program और device के बीच क्या होता है।
3. [कर्नेल लाइब्रेरी](./kernel-library): kernels और उनके configs।
4. [Op Layer](./op-layer): वह API जिसे मॉडल बुलाते हैं।
5. [Tuning](./tuning), [टेस्टिंग और डिबगिंग](./testing-and-debugging), [पोर्टेबिलिटी](./portability)।
