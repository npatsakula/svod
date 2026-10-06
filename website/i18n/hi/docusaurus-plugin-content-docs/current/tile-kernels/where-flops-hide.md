---
sidebar_label: FLOPS कहाँ छिपते हैं
---

# FLOPS कहाँ छिपते हैं

एक आधुनिक AMD matrix core अपने datasheet में जो number छापता है, वह किसी वादे जैसा लगता है — *हज़ारों*
TFLOPS। आप सीधा-सादा matrix multiply लिखते हैं — तीन nested loops, एक multiply, एक add — और जब
measure करते हैं, तो box पर लिखे number का बस चंद percent ही हाथ लगता है।

FLOPS कहीं ग़ायब नहीं हुए। वे छिपे हुए हैं — और उन्हें ढूँढ निकालना ही वह काम है जो *किसी भी* तेज़ matrix
कर्नेल को करना पड़ता है, चाहे उसे compiler generate करे (देखें [अवलोकन](./overview)) या आप उसे हाथ से,
`tk` जैसी किसी tile library के साथ लिखें (जो [HipKittens](https://github.com/HazyResearch/HipKittens) से प्रेरित है)।

यह chapter इस section की बाक़ी हर चीज़ के पीछे का "क्यों" है: यह बताता है कि एक matrix core को
flat-out चलाने के लिए असल में क्या-क्या चाहिए, और वे चंद bottlenecks कौन-से हैं जो रास्ते में आड़े आते हैं।

---

## गणित bottleneck नहीं है

यहाँ बात थोड़ी उलटी लगेगी। matrix multiply *instruction* — जिसे AMD CDNA पर MFMA और RDNA पर WMMA कहता है —
ख़ुद बेहद तेज़ है। एक ही instruction दो छोटे tiles को multiply करके नतीजा accumulate कर देता है। अगर आप
इन्हें लगातार, बिना रुके back-to-back issue कर पाते, तो datasheet वाला number छू लेते।

पर आप ऐसा कर नहीं सकते, क्योंकि हर दो matrix instructions के बीच hardware को यह सब करना पड़ता है:

- अगले tiles के *addresses* compute करना,
- उन्हें ऐसी memory से fetch करना जो math unit के मुक़ाबले कहीं slower है,
- उन्हें registers में *ठीक* उसी layout में लाना जिसकी matrix core को दरकार है,
- और यह सब इस तरह कि math unit load के इंतज़ार में बेकार न बैठा रहे।

> **Roofline intuition.** हर कर्नेल या तो इस बात से सीमित होता है कि वह कितनी तेज़ी से compute कर सकता है
> (compute-bound), या इस बात से कि वह कितनी तेज़ी से data move कर सकता है (memory-bound)। naive matmul
> memory-bound होता है: इसका सारा वक़्त loads के इंतज़ार में बीतता है, और महँगा matrix core यूँ ही idle पड़ा रहता है।
> tiling का मक़सद कर्नेल को compute-bound बनाना है — math unit को feed मिलती रहे।

तो "FLOPS कहाँ जाते हैं?" इस सवाल का असली मतलब है: **matrix instructions को back-to-back issue होने से
आख़िर रोकता क्या है?** इसके पाँच जवाब हैं, जो बार-बार सामने आते हैं।

---

## पाँच gaps

यह framing ThunderKittens और HipKittens पर HazyResearch के काम पर आधारित है (देखें
[HipKittens paper, arXiv:2511.08083](https://arxiv.org/abs/2511.08083))। NVIDIA से AMD पर ideas
port करते वक़्त उनका निष्कर्ष यह रहा कि *tile* और *compute* abstractions तो सीधे चले आते हैं — पर
असली performance जहाँ रहती है, वे हैं **memory, scheduling, और chip layout** से जुड़े फ़ैसले।

### 1. Layout और address computation

matrix core अपने inputs एक ख़ास register layout में चाहता है — कौन-सा element किस lane के पास रहेगा, यह
तय होता है। अगर आपका data ग़लत layout में आता है, तो हर multiply से पहले उसे shuffle करने की कीमत चुकानी पड़ती है।
और *कौन-सा* address load करना है यह compute करना — यानी एक tile coordinate को swizzled buffer में byte offset
में बदलना — ख़ुद एक arithmetic है, जो गणित के साथ होड़ करती है।

इसका हल: tiles को matrix-core fragment के नाप का रखें ताकि data सीधे MMA layout में ही land हो, और
swizzled offsets को हर iteration पर दोबारा compute करने के बजाय एक ही बार *precompute* कर लें।

### 2. Memory latency — और AMD के पास कोई `cp.async` नहीं

NVIDIA पर asynchronous copy instructions (`cp.async`, और बाद में TMA) आपको एक load शुरू करके उसके
land होने तक compute करते रहने देते हैं — `tk` का CUDA path ठीक इसी के लिए `cp.async` इस्तेमाल करता है। AMD
GPUs के पास ये हैं ही नहीं। इसके बदले hardware एक
**सीधे shared memory (LDS) में buffer load** देता है, जो registers को पूरी तरह bypass कर देता है। तेज़ कर्नेल
data का *अगला* block LDS में stream करता रहता है जबकि matrix core *मौजूदा* block पर काम कर रहा होता है।
यहाँ ज़रा-सी चूक हुई, तो math unit हर load पर stall कर जाता है।

### 3. Shared memory में bank conflicts

Shared memory कई banks में बँटी होती है। अगर एक ही wave की दो lanes एक ही cycle में एक ही bank पर hit करें, तो
accesses *serialize* हो जाते हैं — यानी एक memory transaction कई में बदल गया। HipKittens ने
CDNA LDS structure को empirically reverse-engineer किया: **64 banks, और हर एक 32 lanes के दो phases में
access होते हैं।** इसका हल है — in-LDS layout का एक सोच-समझकर चुना गया XOR *swizzle*, ताकि
एक wave की lanes हमेशा अलग-अलग banks में बँट जाएँ। `tk` इन्हीं swizzles को सीधे port करता है।

### 4. Compute और memory को overlap करना

NVIDIA latency को high *occupancy* से छिपाता है — कई warps एक साथ resident रहती हैं, तो एक के stall होने पर
दूसरी चलती रहती है। AMD matrix-core कर्नेल आम तौर पर इस भरोसे नहीं चल सकते, इसलिए वे instruction streams को
interleave करके *explicitly* overlap करते हैं। दो patterns बार-बार दिखते हैं:

- **8-wave ping-pong** — एक producer/consumer बँटवारा, जहाँ कुछ waves सिर्फ़ memory move करती हैं और
  बाक़ी सिर्फ़ compute, और बीच में LDS से होकर लेन-देन होता है।
- **4-wave interleave** — matrix instructions को vector ALU और exponential unit के मुक़ाबले
  finer-grained तरीक़े से interleave करना।

इनमें से कौन जीतेगा, यह *workload पर निर्भर करता है*, कोई तय बात नहीं।

### 5. Chiplet thread-block ordering

एक datacenter AMD GPU कई chiplets (XCDs) से बना होता है, और हर एक के पास L2 cache का अपना हिस्सा होता है।
अगर एक ही data छूने वाली दो workgroups *अलग-अलग* chiplets पर land हो जाएँ, तो वे cache share नहीं कर पातीं। यह
remap करके कि कौन-सा workgroup ID कहाँ run होगा, आप एक-दूसरे के साथ काम करने वाली blocks को एक ही chiplet पर
रखते हैं और असली performance बिना किसी अलग मेहनत के वापस पा लेते हैं।

---

## arch का पहलू: MFMA बनाम WMMA बनाम `mma.sync`, wave32 बनाम wave64

कुछ hardware facts हर उस tile कर्नेल को आकार देते हैं जो `tk` बनाता है, और इन्हें ध्यान में रखना ज़रूरी है:

- **CDNA** (datacenter, gfx942) matrix multiplies को **MFMA** instructions के ज़रिए issue करता है और
  **wave64** चलाता है — प्रति wavefront 64 lanes।
- **RDNA** (RDNA3.5 पर gfx1151, RDNA4 पर gfx1200/gfx1201) **WMMA** instructions issue करता है और
  **wave32** चलाता है — 32 lanes। RDNA3 और RDNA4 आपस में भी मेल नहीं खाते: gfx11 operands को wave के
  दोनों हिस्सों में replicate करता है और accumulator की rows को interleave करता है, gfx12 दोनों छोड़ देता है।
- **NVIDIA** (`sm_80+`) **`mma.sync`** issue करता है और एक **warp32** चलाता है — 32 lanes, पर fragment
  layout फिर से अपना ही: एक 16×16 tile जो दो `m16n8` halves के रूप में रखा जाता है।
- **Apple** (Apple7+) एक 32-lane SIMD group पर 8×8 fragment के साथ `simdgroup_matrix` चलाता है,
  जो बाक़ी सबके fragment का एक-चौथाई है।

lane count बदलते ही यह बदल जाता है कि एक tile के elements wave भर में कैसे बँटते हैं; इससे register layout
बदलता है, और उसके साथ reductions भी — और एक ही width पर भी fragment layout अलग होता है। एक के लिए लिखा कर्नेल
अगर किसी दूसरे पर — इसका हिसाब रखे बिना — चला दिया जाए, तो वह चुपचाप ग़लत नतीजे देता है। एक ही कर्नेल को
इन सब पर correct रखना अपने आप में एक पूरा chapter है:
[Layouts और wave size](./wave-portability)।

:::tip[GPU विशेषज्ञों के लिए]
HipKittens के `analysis/paper_experiments/` micro-benchmarks ऊपर बताए gaps को आँकड़ों में ढालते हैं। यही design को
justify करते हैं:

| Gap | निष्कर्ष |
|-----|---------|
| gap 3 (bank structure) | LDS micro-benchmark पुष्टि करता है कि CDNA पर 64 banks 32 lanes के दो phases में access होते हैं — वही structure जिसके लिए XOR swizzles tuned हैं। |
| gap 4 (overlap हर जगह एक-सा नहीं) | एक BF16 GEMM 8-wave ping-pong के साथ peak करता है; एक FP8 GEMM 4-wave interleave के साथ। इष्टतम wave-overlap रणनीति dtype के अनुसार बदलती है। |
| gap 5 (chiplet swizzle) | XCD locality के लिए workgroup IDs को remap करना एक बड़े GEMM पर एक मापने योग्य speedup देता है। |

`tk` इन levers को सीधे implement करता है: XOR swizzles `tk/src/swizzle.rs` में रहते हैं (HipKittens के
shared-tile layouts से ported, साथ में 16-byte-chunk वाला `Sw16x16Mma` जिसकी `ldmatrix`, `cp.async`
और gfx12 gather को ज़रूरत होती है), L2/chiplet remap `tk/src/grid.rs` में (`l2_swizzle`, जो CUDA GEMM tiles,
सबसे चौड़े RDNA4 tile और gfx942 matmul config में चालू है), और compute/memory overlap को Flash Attention KV
loop पर एक `sched::pipeline(SchedKind::Attention, …)` marker के रूप में व्यक्त किया जाता है, जिसे
`codegen/src/llvm/sched.rs` का एक post-linearization pass consume करता है। आज यह pass CDNA पर दोनों
`SchedKind`s को `@llvm.amdgcn.iglp.opt(0)` में lower करता है — यानी MFMA/memory interleave AMDGPU machine
scheduler को सौंप देता है — और बाक़ी जगह कुछ नहीं करता; gfx942 पर मापा गया नतीजा यह था कि हाथ से रखे
`sched.barrier` fences एक dataflow-scheduled GEMM को अकेले iglp के मुक़ाबले 0.6–0.9× तक *regress* कर देते हैं,
क्योंकि वे ठीक उसी load/MFMA overlap को pin कर देते हैं जिसे पैदा करने के लिए double buffer मौजूद है।

जहाँ fence सचमुच फ़ायदा देता है, वहाँ AUTHOR चेहरा raw controls को `tk/src/asm.rs` में typed `Op::Custom`
nodes के रूप में expose करता है — `s_setprio`, `s_waitcnt_lgkmcnt`, `sched_barrier`, `iglp_opt` — और हर एक किसी
dependency पर पिरोया होता है ताकि वह वहीं land हो जहाँ author ने उसे रखा। tree में इसका इकलौता इस्तेमाल RDNA4 GEMM
है: `ArchCaps::needs_pipeline_commit_fence()` उस arch का नाम बताता है जिसका scheduler pipeline का पूरा LDS commit
trip के MMAs के ऊपर hoist कर देता है, और `gemm_core` इसका जवाब `sched_barrier(0, after_mma)` से देता है ताकि
prefetch matrix work के नीचे in flight बना रहे।
:::

---

## यह क्यों ज़रूरी है

इस section में आगे जो कुछ भी है, वह इन्हीं पाँच gaps में से किसी एक का जवाब है:

- [Tiling क्या है](./tiling), अगला chapter, gaps 1–3 का जवाब देता है: यह data को सही layout में,
  सही memory में, और conflict-free रखता है।
- [Flash Attention](./flash-attention) gaps 2 और 4 को असल काम में दिखाता है: double-buffered streaming
  और एक explicit pipeline।
- [Layouts और wave size](./wave-portability) वह portability tax है जो gap 1, lane-count का फ़र्क़,
  और per-arch fragment layouts आप पर थोपते हैं।

लब्बोलुआब: एक तेज़ GPU कर्नेल बस "गणित, लिख दिया गया" नहीं होता। वह है *गणित, साथ में इस बात का जवाब कि दो
matrix instructions के बीच का हर cycle आख़िर कहाँ जाता है।* FLOPS ठीक वहीं छिपते हैं।

:::note[क्या compiler पहले से matrix cores इस्तेमाल नहीं करता?]
करता है — और यहाँ कुछ भी इसके उलट नहीं कह रहा। graph-native कर्नेल के लिए BEAM का `TC` action एक matmul को
सीधे WMMA/MFMA पर map कर देता है और इसे इन्हीं gaps के मुक़ाबले tile करता है; compiler matrix cores चलाने में
पूरी तरह सक्षम है। ये पाँच gaps वह *hardware reality* हैं जिसे हर तेज़ कर्नेल को हराना होता है —
compiler-generated हो या हाथ से लिखा, फ़र्क़ नहीं — यह compiler में कोई कमी नहीं है।

तो `tk` कोई टक्कर लेने वाला code path नहीं है। यह तो **उन कर्नेल के लिए औज़ार है जिन्हें BEAM express ही नहीं कर सकता**
([अवलोकन](./overview)), और यह अपनी जगह बिना *कोई extra complexity जोड़े* कमाता है: एक `tk` कर्नेल वही
UOp IR emit करता है जो compiler पहले से produce करता है — न कोई दूसरा backend, न अलग toolchain, न कोई नया
debugger। आप इसकी ओर बस तभी हाथ बढ़ाते हैं जब implementation ख़ुद लिखनी हो, और तब भी आप एक ही compiler के
अंदर बने रहते हैं।
:::
