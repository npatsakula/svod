---
sidebar_label: पोर्टेबिलिटी
---

# पोर्टेबिलिटी {#portability}

## Targets {#targets}

`atoms::Target` वह रखता है जो कोई GPU कर सकता है: उसके matrix-core atoms, global memory shared ring
तक कैसे पहुँचती है (`Fill::CpAsync` या `Fill::Staged`), `ldmatrix` और transposing loads, lane moves,
कौन से syncs instructions हैं (`SyncCaps`), occupancy, shared memory और SM count; इसका rustdoc देखें।
`Target::for_device(&spec)` किसी live device से target निकालता है, `Target::for_arch` सिर्फ़ एक
architecture से (जो host tests इस्तेमाल करते हैं), और `Target::offline(arch, sms, xcds)` या
`Target::card(name)` ऐसे कार्ड का वर्णन करता है जिसका device इस मशीन पर नहीं है।

Target यह घोषित करता है कि chip क्या कर सकता है, यह कभी नहीं कि उसकी कीमत क्या है। उसके primitives
की कीमत (`atoms::Costs`) पहले इस्तेमाल पर device पर मापी जाती है (`probe::costs`) और tune store में रखी
जाती है; host tests इसे घोषित करते हैं ([Tuning](./tuning#device-costs))। हर family के लिए कोई cost
tables नहीं हैं। Costs lowering तक `Lowering::device` के रूप में पहुँचती हैं (orientation, exchange या
round trip) और op layer तक `Planner` के device के रूप में (candidate rankings)। मापे गए targets और
उनके आँकड़े [अवलोकन](./overview#targets) में हैं।

`ops::supported(device)` तब true है जब `device` default device हो, उसके target के पास tk kernels हों
(`config::has_kernels`) और वह एक SM count बताता हो; cost probes पहले पूछने पर चलते हैं। किसी भी दूसरे
device पर हर op अपना graph fallback बनाता है (`Fallback::Target`)। Host tests हर atom के operand layouts
को vendor ISA documents के lane formulas के ख़िलाफ़ जाँचते हैं, और kernels वाले हर target के लिए वे हर
kernel family को lower करते हैं, lowering से पहले और बाद में उसे interpret करते हैं, और जहाँ `ptxas` या
clang installed हों वहाँ उनसे compile करते हैं। "Compile हुआ" का मतलब "device पर सही" नहीं है: कोई target
तभी मापा हुआ है जब उसके hardware पर `targets::families_match_the_interpreter_on_the_device` pass हो जाए।

`cp.async` के बिना (AMD के पास कोई asynchronous global → LDS copy नहीं है) pipeline हर step को registers
में load करती है, पिछले step की गणना करती है, फिर registers को shared memory में लिखती है, एक या दो
slots पर। यही path CUDA पर एक device test में चलता है (`register_staged_families_match_on_cuda`)।

## रणनीति {#strategy}

तय किया गया तरीक़ा है **हर op के लिए एक साझा tile program**। `kernels/gemm.rs` जैसा कर्नेल tile values
के ख़िलाफ़ एक बार लिखा जाता है और कभी vendor पर branch नहीं करता। हर target पर जो बदलता है वह data है
जिसे lowering और op layer चुनते हैं:

| हर target पर | कहाँ |
|---|---|
| Matrix-core atoms और उनके layouts | `atoms`, `layout/atoms.rs` |
| Copy के तरीक़े (`cp.async`, `ldmatrix`, register staging, transposing loads) | `Target` fields, emitter |
| Schedule templates | `schedule::Schedule` |
| Config candidate rankings | `ops::config`, मापी गई `Costs` से |

किसी एक vendor के लिए अलग किया गया kernel program सिर्फ़ एक सीमित अपवाद के रूप में अनुमत है।

## इस रिलीज़ में नहीं {#not-in-this-release}

| चीज़ | स्थिति |
|---|---|
| Hopper TMA, mbarrier, warp-specialized producers | शुरू नहीं हुआ; GEMM और attention `cp.async` से भरे जाने वाले `wgmma` चलाते हैं |
| Blackwell `tcgen05`, tensor memory | शुरू नहीं हुआ; sm_120 sm_80 path चलाता है |
| CDNA MFMA 32×32, `buffer_load … lds` | शुरू नहीं हुआ (32×32 layouts मौजूद हैं, atom नहीं) |
| CDNA के बाहर two-crew ping-pong | `Target::crews` सिर्फ़ वहीं 2 है जहाँ matrix core vector units के बगल में चलता है और एक lane की file दो crews के accumulators रखती है (CDNA) |
| Apple | सिर्फ़ atoms; कोई kernels नहीं |
| Convolution का image-staged रूप, depthwise, grouped | ग्राफ़ |
| int8 | Kernels वाले हर target पर atoms (`m16n8k32` s8, RDNA `iu8`, CDNA `16x16x32` i8, i32 में); nearest-neighbour कर्नेल और `k_nearest_reranked` इन्हें इस्तेमाल करते हैं, GEMM और attention ops नहीं |
| fp8, attention backward | शुरू नहीं हुआ |
