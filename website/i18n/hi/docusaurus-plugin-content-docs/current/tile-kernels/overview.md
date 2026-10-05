---
sidebar_label: अवलोकन
---

# हाथ से लिखे कर्नेल क्यों?

Svod की पूरी बुनियाद ऑटोमेशन पर है। आप एक lazy ग्राफ़ बनाते हैं, `realize()` कॉल करते हैं, और हर लूप को
कैसे tile, vectorize और parallelize करना है, यह ऑप्टिमाइज़र ख़ुद तय कर लेता है — [beam search](../architecture/optimizations/kernel-search)
के साथ तो यह सैकड़ों candidate schedules को compile करके उनका time तक माप लेगा, ताकि सबसे तेज़ वाला चुना जा सके। आपको एक भी लूप ख़ुद नहीं लिखना पड़ता।

तो फिर Svod एक पूरा crate — `svod-tk`, छोटे में `tk` — क्यों रखता है, जिसका इकलौता काम आपको
GPU कर्नेल हाथ से लिखने देना है?

वजह यह है कि कुछ कर्नेल loop transformations पर सर्च करके खोजे ही नहीं जा सकते। ऑप्टिमाइज़र का
action space बस इतना है — "इस reduction को लो, इसे tile करो, unroll करो, इसे shared memory में डालो।"
यह matmul के लिए काफ़ी है, fused feed-forward block के लिए काफ़ी है, layernorm के लिए काफ़ी है। पर
Flash Attention के लिए **काफ़ी नहीं**, क्योंकि इसका गणित एक *recurrence* है: keys का हर block एक
running maximum और एक running sum को अपडेट करता है, और साथ-साथ accumulator को rescale करता रहता है। यहाँ tile करने लायक़ कोई एक
`REDUCE` है ही नहीं — loop body हर बार पिछले iteration के नतीजे पर निर्भर करती है। आप जितनी मर्ज़ी
axis-shuffling कर लें, यह उससे नहीं निकलती।

ऐसे कर्नेल के लिए आपको algorithm ख़ुद लिखना पड़ता है। `tk` वही तरीक़ा है जिससे आप यह करते हैं — और वह भी
compiler से बाहर निकले बिना।

---

## `tk` एक builder है, backend नहीं

जब हाथ से लिखे कर्नेल की ज़रूरत पड़ती है, तो आसान रास्ता यही लगता है कि एक अलग code path जोड़ दिया जाए: एक
छोटा-सा GPU DSL जो ख़ुद की assembly emit करे और बाक़ी सब चीज़ों से अलग launch हो। पर अब आपके पास
दो compilers हो गए, दो debuggers, और दो mental models।

`tk` यही करने से इनकार करता है। इसके अपने शब्दों में, यह *"एक thin eager builder है, backend नहीं।"* जब आप
`tk` से कोई कर्नेल author करते हैं, तो यह machine code emit नहीं करता — यह **वही UOp IR** emit करता है जो
बाक़ी Svod पहले से बोलता है: explicit `RANGE` loops, `INDEX`/`STORE` memory ops, और `WMMA`
matrix-core ops। यानी बिल्कुल वही intermediate representation जिसका वर्णन
[एक IR सबके लिए](../architecture/ir-design) में है।

इसका मतलब यह हुआ कि हाथ से लिखा `tk` कर्नेल और autotuned graph कर्नेल — दोनों *एक ही तरह की object* हैं।
एक ही UOp DAG के दो subgraphs, जिन्हें एक ही renderer render करता है और एक ही runtime चलाता है।
[IR में authoring](./lowering) में ठीक-ठीक दिखाया गया है कि यह कैसे होता है।

---

## `tk` के तीन चेहरे

आप कौन हैं और क्या कर रहे हैं, इसके हिसाब से `tk` तीन में से कोई एक interface सामने रखता है (तीनों ही
`tk/src/lib.rs` से re-export होते हैं):

| चेहरा | आप हैं… | आप किससे काम लेते हैं |
|------|----------|----------------|
| **USE** | एक application author जिसे बस एक तेज़ कर्नेल चाहिए | `flash_attention`, अपने fused `Epilogue`s के साथ `gemm_nt`, `rms_norm` / `add_rms_norm`, `single_query_attention`, `knn`, `kmeans_assign`, और square `matmul` — ये lazy `Tensor` लौटाते हैं, कर्नेल की कोई जानकारी ज़रूरी नहीं |
| **AUTHOR** | एक नया tile कर्नेल लिख रहे | `Kernel` / `Group` / `Loop` builder, `ArchCaps` और `FragRole`, tile types (`GL`/`ST`/`RT`/`RV`), `MoveIdx`, `graph_launch` और `launch_custom` |
| **DEBUG** | किसी कर्नेल को isolation में test या benchmark कर रहे | `run_kernel`, `compile_kernel`, `CompiledLaunch`, structural `KernelFingerprint`s, और `tune` store |

ज़्यादातर पाठकों के लिए असली काम का चेहरा USE ही है: `flash_attention(q, k, v)` आपको वापस
एक आम-सा `Tensor` देता है, जो बाक़ी हर tensor की तरह lazy graph में शामिल हो जाता है। आपको tile नज़र तक नहीं आता।
[कर्नेल लाइब्रेरी](./kernel-library) हर कर्नेल की सूची देती है; [Tiling क्या है](./tiling) AUTHOR चेहरा
खोलता है; [डीबगिंग](./debugging) DEBUG को कवर करता है।

---

## Targets

एक `tk` कर्नेल किसी matrix-core family के लिए बनता है, और हर कर्नेल उन parts को, जिन पर उसे validate
किया गया है, एक `ArchSet` (`tk/src/target.rs`) के रूप में घोषित करता है। किसी दूसरे device पर उसका launcher
`Ok(None)` लौटाता है और caller वापस graph पर fall back कर जाता है।

| Family | Parts | Matrix op | Wave |
|---|---|---|---|
| AMD CDNA3 | gfx942 | MFMA | 64 lanes |
| AMD RDNA3.5 | gfx1151 | gfx11 WMMA | 32 lanes |
| AMD RDNA4 | gfx1200, gfx1201 | gfx12 WMMA | 32 lanes |
| NVIDIA | sm_80 और नए | `mma.sync.m16n8k16` | 32 lanes |
| Apple | Apple7 और नए (सिर्फ़ flash attention और `matmul`) | `simdgroup_matrix` 8×8 | 32 lanes |

Inputs bf16 या f16 होते हैं, accumulation f32 में। कौन-सा कर्नेल कहाँ चलता है, यह
[कर्नेल लाइब्रेरी](./kernel-library) की table में है; एक ही body इन सब पर क्यों चलती है, यह
[Layouts और wave size](./wave-portability) में।

---

## हाथ से कब लिखें, और कब BEAM पर छोड़ें

इसका बस एक नियम है, और वह सीधे इस बात से निकलता है कि *BEAM असल में किस चीज़ पर सर्च करता है*।

BEAM — और वह heuristic optimizer जिस पर यह fall back करता है — किसी *fixed* computation के लिए **schedules** का
space सर्च करते हैं। किसी कर्नेल का dataflow graph मिलने पर वे इसे tile, vectorize, unroll और
parallelize करने, shared memory से होकर stage करने, और matrix cores पर map करने के तरीक़े आज़माते हैं (`OptOps` actions:
`UPCAST`, `UNROLL`, `LOCAL`, `GROUP`, `TC`, …)। पर एक चीज़ वे कभी नहीं करते — यह बदलना कि *क्या* compute होगा।
graph के nodes — adds, muls, और reductions — तय हैं; बस उनकी arrangement खुली है।

तो बात यह है:

> अगर किसी कर्नेल को बस एक fixed dataflow का अच्छा **schedule** चाहिए, तो इसे BEAM पर छोड़ दें। पर अगर इसे
> naive वाले से एक **अलग ही algorithm** चाहिए — कुछ ऐसा जो मौजूदा ops की किसी भी reordering से नहीं निकलता — या
> किसी **kernel boundary के आर-पार fusion** चाहिए जिसे scheduler बनाए रखता है, तो आपको इसे ख़ुद लिखना होगा।

| कर्नेल का स्वभाव | किससे बनता है | उदाहरण |
|------------------------|----------|----------|
| **Fixed dataflow** — एक rectangular iteration space पर elementwise ops और reductions; बस *schedule* (tiling, vectorization, data placement, matrix-core mapping) खुला है | graph ops + **BEAM** | एक साधारण matmul, feed-forward block, layernorm, softmax |
| **एक नए सिरे से गढ़ा गया algorithm चाहिए** — एक loop-carried recurrence, या ऐसे restructured numerics जिन्हें naive ops का कोई reschedule नहीं निकाल सकता | **`tk` में हाथ से authored** | Flash Attention (online softmax); single-query attention (streamed cache पर one-pass softmax); k-NN और k-means assignment — streamed tiles पर एक running top-K / argmin के साथ fuse किया गया एक cross-term WMMA, ताकि पूरी `[N, M]` distance matrix कभी बने ही नहीं |
| **ऐसा fusion चाहिए जिसे scheduler express नहीं कर सकता** — matrix-core कर्नेल के अपने store में मोड़ा गया काम, या एक बार में लिखा गया residual stream | **`tk` में हाथ से authored** | residual-add या SwiGLU `Epilogue` के साथ `gemm_nt`; `add_rms_norm` |

### BEAM जहाँ नहीं पहुँच सकता

Naive attention पूरी `N×N` score matrix बनाता है, उस पर एक global softmax लेता है, फिर
`V` से multiply करता है। BEAM इसे tile और vectorize ज़रूर कर सकता है, पर तब भी यह पूरी
score matrix materialize कर ही देगा — और यही वह cost है जिससे बचने के लिए Flash Attention मौजूद है।

तेज़ वाला version वह matrix बनाता ही नहीं। यह keys के blocks पर stream करता है, एक running max
और sum रखता जाता है, और हर block के आने पर output को rescale करता रहता है: यही online softmax है। यह naive
computation का reschedule नहीं है; यह एक अलग dataflow है जिसमें loop-carried dependency है — हर
block वही state पढ़ता है जो पिछले block ने लिखा। कोई भी `UPCAST`/`UNROLL`/`TC` sequence एक recurrence
पैदा नहीं कर सकती, इसलिए online softmax BEAM के search space से बाहर ही रहता है। यह फ़ासला algorithm का है, schedule का
नहीं — और इसी फ़ासले को `tk` भरता है।

तीसरी row एक संकरा फ़ासला है। graph की linear layer एक matrix-core कर्नेल है जिसे BEAM अच्छी तरह tune करता है,
पर उसके बाद का residual add, या fused gate/up projection के बाद आने वाला `silu(gate) · up`, output पर एक दूसरा
pass है जिसे scheduler GEMM के epilogue में नहीं मोड़ता। `gemm_nt` इसे मोड़ देता है, इसलिए intermediate कभी
लिखा ही नहीं जाता। पहली row का square `matmul` फिर अलग बात है: यह DSL के लिए एक performance canary है,
production वाला matmul नहीं — production वाला तो graph से होकर जाता है।

:::tip[GPU विशेषज्ञों के लिए]
हाथ से authored कर्नेल और BEAM-tuned कर्नेल के बीच structural अंतर बस इतना-सा है — `SINK` UOp के
`KernelInfo` पर एक अकेला field: graph कर्नेल `opts_to_apply: None` छोड़ देता है, जबकि `tk` कर्नेल इसे
`Some(vec![])` सेट करता है। IR वही, pipeline वही, बस एक marker का फ़र्क़। [IR में authoring](./lowering) इसे
end to end ट्रेस करता है।
:::

---

## यह section आगे कहाँ जाता है

बाक़ी का section hardware की समस्या से शुरू होकर design की तुलना तक पहुँचता है:

1. **[FLOPS कहाँ छिपते हैं](./where-flops-hide)** — एक matrix core को saturate करना इतना मुश्किल क्यों है,
   और वे चंद bottlenecks जिन्हें हर तेज़ कर्नेल को हराना पड़ता है।
2. **[Tiling क्या है](./tiling)** — वह abstraction जो उन bottlenecks का जवाब देता है, और `tk`
   type system में tiles को कैसे represent करता है।
3. **[IR में authoring](./lowering)** — एक `tk` कर्नेल कैसे UOps में बदलता है और
   lazy graph में शामिल होता है।
4. **[एक कर्नेल लिखना](./first-kernel)** — सबसे सरल कर्नेल को step by step author और run करना।
5. **[Builder API](./builder-reference)** — AUTHOR चेहरे का हर type और method।
6. **[Layouts और wave size](./wave-portability)** — एक ही कर्नेल को पाँच fragment layouts और
   दो wave चौड़ाइयों पर correct रखना।
7. **[कर्नेल लाइब्रेरी](./kernel-library)** — shipped कर्नेल, उनके contracts, और कोई model उन्हें
   कैसे इस्तेमाल करता है।
8. **[Flash Attention](./flash-attention)** — वह worked example जिसने इस सबकी नींव रखी।
9. **[Autotuning](./tuning)** — पहली बार इस्तेमाल पर किसी कर्नेल का tile कैसे मापा और cache किया जाता है।
10. **[डीबगिंग](./debugging)** — कर्नेल को हाथ से run और verify करना।
11. **[Profiling और Benchmarking](./profiling)** — किसी भी `Tensor` या `ExecutionPlan` के लिए, layered
    profiler और criterion integration।
12. **[tk बनाम HipKittens बनाम CuTile](./comparison)** — यह design इस पूरे landscape में कहाँ बैठता है।
