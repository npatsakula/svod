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

| Target | Atoms और layouts | Config tables | Lower होकर चला |
|---|---|---|---|
| CUDA sm_80+ (sm_86 पर मापा गया) | `mma.sync` m16n8k16, `ldmatrix`, `cp.async` | हाँ | हाँ |
| AMD CDNA (gfx942) | MFMA 16×16×16 | नहीं | नहीं |
| AMD RDNA3 / RDNA4 | WMMA 16×16×16 | नहीं | नहीं |
| Apple | simdgroup 8×8×8 | नहीं | नहीं |

`ops::supported(device)` सिर्फ़ वहीं true है जहाँ config tables मौजूद हैं, इसलिए हर दूसरे device पर हर
op अपना graph fallback बनाता है (`Fallback::Target`)। AMD और Apple atoms मौजूद हैं, और host tests
जाँचते हैं कि उनके layouts instruction shape को tile करते हैं, पर उन targets पर कोई कर्नेल lower या
run नहीं किया गया है।

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
| Hopper (sm_90a): wgmma, TMA, mbarrier, warp-specialized template | शुरू नहीं हुआ; remote H100 hardware पर मापा जाना है |
| Blackwell (sm_100a): tcgen05, tensor memory | शुरू नहीं हुआ; B200 तक पहुँच नहीं |
| AMD CDNA: ping-pong template, MFMA 32×32, `buffer_load … lds` | शुरू नहीं हुआ; remote MI300X hardware पर मापा जाना है |
| AMD RDNA, Apple | सिर्फ़ atoms; न tables, न emit किए गए कर्नेल |
| Warp roles, role barriers, raw asm statements | IR में रिकॉर्ड होते हैं, lowering उन्हें ठुकराती है |
| Convolution (implicit GEMM) | tk3 में शुरू नहीं हुआ; YOLO का convolution tk1 पर चलता है |
| fp8/int8 weights, GEMV + argmax, persistent grid, attention backward | शुरू नहीं हुआ |
