---
sidebar_label: अवलोकन
---

# Tile कर्नेल (tk3)

Svod का ऑप्टिमाइज़र किसी मॉडल के ज़्यादातर हिस्से के लिए तेज़ schedules loop transformations पर सर्च
करके खोज लेता है। कुछ कर्नेल इस तरह नहीं मिलते। Flash attention एक recurrence है: keys का हर block
एक running maximum और sum को अपडेट करता है और accumulator को rescale करता है, इसलिए tile करने के लिए
कोई एक reduction है ही नहीं। Tensor-core GPU पर तेज़ GEMM एक multi-stage `cp.async` ring,
`ldmatrix` fragment loads और swizzled shared layout पर टिका होता है, और ये loop search के कदम नहीं
हैं। ऐसे कर्नेल हाथ से लिखे जाते हैं, और `svod-tk3` ("tk3") उन्हें लिखने का crate है।

tk3 पुराने `svod-tk` crate (tk1) का उत्तराधिकारी है। Transformer मॉडल अपने hand kernels यहाँ बताई
गई op layer पर चलाते हैं; YOLO का convolution (एक implicit GEMM) `svod-tk` पर चलता है।

## tk3 क्या है {#what-tk3-is}

| हिस्सा | मॉड्यूल | क्या करता है |
|---|---|---|
| Tile program | `ir`, `build` | tile values पर एक structured statement tree (`Let`, `Copy`, `Loop`, `Pipeline`, `If`)। इसे एक recording `Kernel` builder program order में बनाता है, और यह DAG नहीं है। |
| Layouts | `layout`, `layouts` | हर fragment और swizzle के लिए F2 linear layouts (bit matrices), जिन्हें inference तय करता है। Kernel code कभी किसी lane का नाम नहीं लेता। |
| Atoms | `atoms` | हर target के matrix-core instructions जो अपने operand layouts साथ लाते हैं (CUDA पर `mma.sync`)। |
| Schedule templates | `schedule` | एक `Pipeline` statement को loops, `cp.async` groups और barriers में फैलाते हैं। लेखक सिर्फ़ produce और consume bodies लिखता है। |
| Lowering | `lower` | Schedule को फैलाता है, effects से barriers डालता है, layouts infer करता है और एक pre-linearized instruction list (`Op::Linear`) emit करता है, ताकि क्रम कोई toposort तय न करे। |
| Kernels | `kernels` | Epilogues वाला GEMM, flash attention, attention prologue, LayerNorm/RMSNorm। हर एक में spec struct जाता है और `Program` निकलता है। |
| Launch | `launch` | किसी program को lazy ग्राफ़ में custom kernel की तरह चलाता है, lowered bodies को memoize करते हुए। |
| Op layer | `ops` | `linear`, `attention`, `heads`, `layer_norm`, `rms_norm`, … ये हमेशा एक `Tensor` लौटाते हैं: kernel फ़िट हो तो kernel, वरना ग्राफ़। |
| Interpreter | `interp` | किसी भी tile program को host पर चलाकर बिना GPU के reference numerics देता है। |
| Tune store | `tune` | हर op के config candidates को हर shape और device पर एक बार मापता है और विजेता को डिस्क पर रखता है। |

```text
model code ──► svod_tk3::ops          always a Tensor; picks a kernel and config or builds the graph op
                   │
                   ▼
              kernels::*               spec ─► tile Program (builder, program order)
                   │
                   ▼
              lower::lower             schedule template ─► operand loads ─► barriers
                   │                   ─► layout inference ─► emission
                   ▼
              Op::Linear program  ──►  custom kernel in the lazy graph ─► LLVM ─► device
```

## स्थिति, RTX 3060 (sm_86) पर मापी गई {#status-measured-on-an-rtx-3060-sm_86}

| क्या | नतीजा |
|---|---|
| bf16 GEMM 4096³ | 25.4 TFLOP/s, tk1 के बराबर। इस कार्ड की `mma.sync` bf16→f32 सीमा 27.9 है |
| Flash attention forward | हर throughput probe पर tk1 के बराबर या उससे ऊपर (head dims 64 और 128, causal और non-causal) |
| LayerNorm / RMSNorm | memory bandwidth का 92% |
| Nemotron-3-Diarization (bf16) | चार steps पर 68.2 → 49–53 ms, 2292 → 1304 dispatches, parity बरक़रार |

Nemotron अपने सभी projections, norms और attention के लिए op layer इस्तेमाल करता है। Whisper
encoder, ModernBERT, GigaAM, XLM-R/BGE-M3 और Qwen3 इसे projections और norms के लिए बुलाते हैं।

:::note[अभी पूरा नहीं]
मॉडलों में decode attention, convolution kernels, और NVIDIA sm_80+ के अलावा कोई भी target
(Hopper, Blackwell और AMD CDNA/RDNA paths) implement नहीं हुए हैं। देखें
[पोर्टेबिलिटी](./portability)।
:::

## पढ़ने का क्रम {#reading-order}

1. [आपका पहला कर्नेल](./first-kernel): builder, GEMM के सहारे।
2. [Layouts और Lowering](./layouts-and-lowering): program और device के बीच क्या होता है।
3. [कर्नेल लाइब्रेरी](./kernel-library): kernels और उनके configs।
4. [Op Layer](./op-layer): वह API जिसे मॉडल बुलाते हैं।
5. [Tuning](./tuning), [टेस्टिंग और डिबगिंग](./testing-and-debugging), [पोर्टेबिलिटी](./portability)।
