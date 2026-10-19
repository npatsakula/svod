---
sidebar_label: अवलोकन
---

# Tile कर्नेल (tk3) {#tile-kernels-tk3}

Svod का ऑप्टिमाइज़र किसी मॉडल के ज़्यादातर हिस्से के लिए तेज़ schedules loop transformations पर सर्च
करके खोज लेता है। कुछ कर्नेल इस तरह नहीं मिलते। Flash attention एक recurrence है: keys का हर block
एक running maximum और sum को अपडेट करता है और accumulator को rescale करता है, इसलिए tile करने के लिए
कोई एक reduction है ही नहीं। Tensor-core GPU पर तेज़ GEMM एक multi-stage `cp.async` ring,
`ldmatrix` fragment loads और swizzled shared layout पर टिका होता है, और ये loop search के कदम नहीं
हैं। ऐसे कर्नेल हाथ से लिखे जाते हैं, और `svod-tk3` ("tk3") उन्हें लिखने का crate है।

मॉडलों के hand kernels के स्रोत के रूप में tk3 ने tk1 (`svod-tk`) की जगह ले ली है; अब कोई मॉडल tk1 को
नहीं बुलाता, और `tk/` tree में बना रहता है। आठ मॉडल op layer पर चलते हैं: Nemotron-3-Diarization,
Whisper (encoder, और cache mode में decoder step का attention), ModernBERT,
BGE-M3/XLM-RoBERTa, Qwen3-Embedding, GigaAM, WavLM और YOLO26 (channels-last, जिसके 3×3
convolutions `ops::conv2d` पर चलते हैं)।

## tk3 क्या है {#what-tk3-is}

| हिस्सा | मॉड्यूल | क्या करता है |
|---|---|---|
| Tile program | `ir`, `build` | tile values पर एक structured statement tree (`Let`, `Copy`, `Loop`, `Pipeline`, `If`, `Warps`), जिसे एक `Kernel` builder program order में रिकॉर्ड करता है। यह DAG नहीं है। |
| Layouts | `layout`, `layouts` | हर fragment और swizzle के लिए F2 linear layouts (bit matrices), जिन्हें inference तय करता है। Kernel code कभी किसी lane का नाम नहीं लेता। |
| Atoms | `atoms` | हर target के matrix-core instructions अपने operand layouts के साथ: `mma.sync` (CUDA), `wgmma` (sm_90), WMMA (RDNA3/3.5/4), MFMA 16×16×16 (CDNA)। |
| Schedule templates | `schedule` | एक `Pipeline` को `cp.async` groups, एक या दो slots वाली register-staged ring, या ping-pong में दो crews (CDNA GEMM) में फैलाते हैं। लेखक सिर्फ़ produce और consume bodies लिखता है। |
| Lowering | `lower` | Effects से barriers डालता है, schedule को फैलाता है, layouts infer करता है और एक pre-linearized instruction list (`Op::Linear`) emit करता है, ताकि क्रम कोई toposort तय न करे। |
| Kernels | `kernels` | Epilogues वाला GEMM, split-K के साथ implicit-GEMM 3×3 convolution, flash attention, attention prologue, LayerNorm/RMSNorm। हर एक में spec struct जाता है और `Program` निकलता है। |
| Launch | `launch` | किसी program को lazy ग्राफ़ में custom kernel की तरह चलाता है, lowered bodies को memoize करते हुए। |
| Op layer | `ops` | `linear`, `conv2d`, `attention`, `heads`, `layer_norm`, `rms_norm`, … ये हमेशा एक `Tensor` लौटाते हैं: kernel फ़िट हो तो kernel, वरना ग्राफ़। |
| Planner | `ops::config` | हर shape के configs को device की मापी गई costs से रैंक करता है। |
| Interpreter | `interp` | किसी भी tile program को host पर चलाकर बिना GPU के reference numerics देता है। |
| Tune store | `tune` | हर op के candidates को हर shape और device पर एक बार मापता है और विजेता को डिस्क पर रखता है। |

```text
model code ──► svod_tk3::ops          always a Tensor; picks a kernel and config or builds the graph op
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

| Target | Matrix core | Fills | किस पर मापा गया | GEMM 4096³ bf16, TFLOP/s | Attention d64, TFLOP/s |
|---|---|---|---|---|---|
| CUDA sm_80+ | `mma.sync` m16n8k16, `ldmatrix` | `cp.async` | RTX 3060 (sm_86); RTX 5070, RTX 5090 (sm_120) | 25.9; 65.7 (cuBLAS NN 67.4); 204.8 (cuBLAS NN 218.2) | 24.8; 62.9; 188.5 |
| CUDA sm_90 | `wgmma` जोड़ता है (GEMM और attention) | `cp.async` | H100 PCIe | 423.7 (cuBLASLt 410–524) | 254.9 (FA3 270.1) |
| AMD RDNA4 | WMMA 16×16×16 | registers, 1–2 slots | RX 9070 XT (gfx1201) | 148.8 (rocBLAS 80) | 117.3 |
| AMD RDNA3 / 3.5 | WMMA 16×16×16 | registers, 1–2 slots | Strix Halo (gfx1151); gfx1100 compile हुआ, चलाया नहीं गया | 29.7 (rocBLAS NT 32.5) | 26.2 |
| AMD CDNA3 / 4 | MFMA 16×16×16, wave64 | registers, ping-pong में दो crews | MI300X (gfx942); gfx950 चलाया नहीं गया | 560.0 (hipBLASLt 615) | 281.4 |
| Apple | simdgroup 8×8×8 atom | — | कोई kernels नहीं | — | — |

हर आँकड़ा उस कार्ड पर tuner का चुनाव है, जिसे 2026-10-17 को `gemm_throughput_probe` और
`attention_throughput_probe` (B 4, H 8, T 2048) से मापा गया, commit 55fb615b पर, सिवाय H100
(251276d9) और MI300X (fc5cc954) के। कोष्ठक में दिए vendor references उसी मशीन पर मापे गए। कोई target
तभी मापा हुआ गिना जाता है जब उस पर `targets::families_match_the_interpreter_on_the_device` pass हो जाए।

RTX 3060 पर 8192×1024 bf16 पर RMSNorm कार्ड के 360 में से 325 GB/s ले जाता है, और
Nemotron-3-Diarization (bf16, चार steps) 1064 dispatches में 45.82 ms GPU time लेता है, जबकि tk3 से
पहले 2292 dispatches थे।

:::note[इस रिलीज़ में नहीं]
Hopper TMA और mbarrier producers (sm_90 `cp.async` से भरे जाने वाले `wgmma` चलाता है), Blackwell
`tcgen05`, 32×32 MFMA, Apple kernels, fp8/int8, attention backward, और convolution का image-staged
रूप। देखें [पोर्टेबिलिटी](./portability#not-in-this-release)।
:::

## पढ़ने का क्रम {#reading-order}

1. [आपका पहला कर्नेल](./first-kernel): builder, GEMM के सहारे।
2. [Layouts और Lowering](./layouts-and-lowering): program और device के बीच क्या होता है।
3. [कर्नेल लाइब्रेरी](./kernel-library): kernels और उनके configs।
4. [Op Layer](./op-layer): वह API जिसे मॉडल बुलाते हैं।
5. [Tuning](./tuning), [टेस्टिंग और डिबगिंग](./testing-and-debugging), [पोर्टेबिलिटी](./portability)।
