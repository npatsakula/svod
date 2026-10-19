---
sidebar_label: पोर्टेबिलिटी
---

# पोर्टेबिलिटी

## Targets {#targets}

`atoms::Target` वह सब है जो lowering किसी GPU के बारे में जानती है:

| Field | मतलब |
|---|---|
| `arch` | `GpuArch::{Cuda, Amd, Metal}` |
| `wave` | हर warp या wave में lanes |
| `mma` | अपने operand layouts के साथ matrix-core atoms |
| `cp_async` | Asynchronous global → shared copies (CUDA sm_80+) |
| `ldmatrix` | Warp-collective 8×8 b16 fragment loads (CUDA sm_75+) |
| `smem_bytes` | हर block की shared memory (device बताए तो opt-in सीमा) |
| `sms` | SM या CU गिनती, जब device बताए |

`Target::for_device(&spec)` एक live device से target निकालता है। `Target::for_arch(arch)` सिर्फ़
architecture से एक target बनाता है, जिसे host tests इस्तेमाल करते हैं। `atoms::sm86()` 28 SMs वाला
RTX 3060 target है।

| Target | Atoms | Fills | Tables | स्थिति |
|---|---|---|---|---|
| CUDA sm_80+ (sm_86) | `mma.sync` m16n8k16, `ldmatrix` | `cp.async`, 2–3 stages | हाँ | RTX 3060 पर मापा गया |
| CUDA sm_90 (Hopper) | `mma.sync` m16n8k16, `ldmatrix` | `cp.async`, 227 KB तक shared | sm_80 tables | Compile हुआ: हर family `ptxas -arch=sm_90` और `sm_90a` से assemble होती है; चलाया नहीं गया |
| AMD RDNA4 (gfx1200, gfx1201) | WMMA 16×16×16, हर lane में 8 values | register-staged, 2 stages | हाँ, मापी नहीं गईं | Code objects में compile हुआ; चलाया नहीं गया |
| AMD RDNA3 / RDNA3.5 (gfx1100–1102, gfx1151) | WMMA 16×16×16, replicated inputs | register-staged, 2 stages | हाँ, मापी नहीं गईं | Code objects में compile हुआ; चलाया नहीं गया |
| AMD CDNA3 / CDNA4 (gfx942, gfx950) | MFMA 16×16×16, wave64 | register-staged, 2 stages | हाँ, मापी नहीं गईं | gfx942 के लिए compile हुआ; चलाया नहीं गया |
| Apple | simdgroup 8×8×8 | कोई नहीं | नहीं | सिर्फ़ atoms |

`ops::supported(device)` वहीं true है जहाँ config tables हैं; किसी और device पर हर op अपना
graph fallback बनाता है (`Fallback::Target`)। Host tests हर atom के operand layouts को vendor
ISA documents के lane formulas से मिलाते हैं, और tables वाले हर target के लिए हर kernel family
को lower करते हैं, lowering से पहले और बाद interpret करते हैं, और `ptxas` या clang installed
हों तो compile करते हैं। "Compile हुआ" का मतलब "device पर सही" नहीं है: कोई target तभी मापा
हुआ माना जाता है जब उसके hardware पर `targets::families_match_the_interpreter_on_the_device`
pass हो।

`cp.async` के बिना (RDNA में global → LDS copy नहीं है) pipeline हर step को registers में load
करती है, पिछले step की गणना करती है, फिर registers को shared memory में लिखती है, दो slots पर।
यही path CUDA पर एक device test में चलता है (`register_staged_families_match_on_cuda`)।

## रणनीति {#strategy}

तय किया गया तरीक़ा है **हर op के लिए एक साझा tile program**। `kernels/gemm.rs` जैसा कर्नेल tile values
के ख़िलाफ़ एक बार लिखा जाता है और कभी vendor पर branch नहीं करता। हर target के साथ जो बदलता है वह
data है, जिसे lowering और op layer चुनते हैं:

| हर target के लिए | कहाँ |
|---|---|
| Matrix-core atoms और उनके layouts | `atoms`, `layout/atoms.rs` |
| Copy mechanisms (`cp.async`, `ldmatrix`, register staging) | `Target` flags, emitter |
| Schedule templates | `schedule::Schedule` |
| Config candidate tables | `ops::config` |

किसी एक vendor के लिए fork किया गया कर्नेल program सिर्फ़ एक सीमित अपवाद के रूप में मान्य है।

## अभी पूरा नहीं {#not-done}

| आइटम | स्थिति |
|---|---|
| Hopper (sm_90a): wgmma, TMA, mbarrier, warp-specialized template | शुरू नहीं हुआ; Hopper `mma.sync` path पर चलता है (substrate items 5–8) |
| Blackwell (sm_100a): tcgen05, tensor memory | शुरू नहीं हुआ; B200 तक पहुँच नहीं |
| AMD CDNA: ping-pong template, MFMA 32×32, `buffer_load … lds` | शुरू नहीं हुआ |
| AMD पर माप | कोई AMD kernel नहीं चला; tables पहले अनुमान हैं, जिनमें से tune store चुनता है |
| AMD attention registers | Score tile shared memory से होकर P·V तक पहुँचता है, और V operand element by element इकट्ठा होता है; RDNA पर d = 128 spill करता है |
| Apple | सिर्फ़ atoms; न tables, न emit किए गए कर्नेल |
| Warp roles, role barriers, raw asm statements | IR में रिकॉर्ड होते हैं, lowering उन्हें ठुकराती है |
| Convolution (implicit GEMM) | sm_86 पर मापा गया, जहाँ YOLO26 इस पर channels-last चलता है (`ops::conv2d`); AMD के लिए सिर्फ़ compile हुआ |
| fp8/int8 weights, GEMV + argmax, persistent grid, attention backward | शुरू नहीं हुआ |
