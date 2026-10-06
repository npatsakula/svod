---
sidebar_label: Layouts और wave size
---

# एक कर्नेल को पाँच layouts पर correct रखना

यह रहा एक ऐसा bug जो NVIDIA hardware आपको दे ही नहीं सकता। आप एक tile कर्नेल लिखते हैं, इसे एक CDNA
datacenter GPU पर test करते हैं, और यह बिल्कुल सही चलता है। फिर *वही* कर्नेल आप एक RDNA laptop APU पर run
करते हैं और numbers बिल्कुल कचरा निकलते हैं — न कोई crash, न कोई error, बस ग़लत। code में देखने से कुछ
अलग नहीं लगता। CUDA इस ख़ास जाल से बचा रहता है — warp हर जगह 32 lanes का होता है — पर इसके पीछे की वजह से
नहीं: fragment layout तब भी अलग होता है, इसलिए वही indirection कर्नेल को NVIDIA पर, और Apple पर भी, ले जाती है।

[Tiling क्या है](./tiling) ने fragments और role-based selection से परिचय कराया था; यह chapter बताता है कि
उस indirection का होना आख़िर ज़रूरी क्यों है। असली दोषी है **wavefront size** और, उसके नीचे, किसी fragment का
**lane map**; इन दोनों से साफ़-सुथरे ढंग से निपटना ही वह बात है जो एक chip पर चलने वाली tile library को एक
सचमुच portable library से अलग करती है।

---

## 32-बनाम-64 का बँटवारा

एक wavefront (NVIDIA का "warp", Apple का "SIMD group") उन lanes का समूह है जो lockstep में execute होती हैं।
AMD पर इसके दो sizes हैं, और Svod दोनों को target करता है, साथ ही NVIDIA और Apple वालों को भी:

| Family | Parts | Matrix op | Wavefront | Fragment | प्रति lane |
|---|---|---|---|---|---|
| **CDNA3** | gfx942 | MFMA | 64 | 16×16, हर role के लिए `Strided { stride: 4 }` | 4 |
| **RDNA3** | gfx1151 | gfx11 WMMA | 32 | 16×16, `Interleaved` accumulator, wave के दोनों हिस्सों में replicated `Strided { stride: 0 }` operand | 8 acc / 16 operand |
| **RDNA4** | gfx1200, gfx1201 | gfx12 WMMA | 32 | 16×16, हर role के लिए `Strided { stride: 8 }` | 8 |
| **CUDA** | sm_80+ | `mma.sync.m16n8k16` | 32 | दो `m16n8` halves के रूप में 16×16, हर role के लिए `MmaSync` | 8 |
| **Metal** | Apple7+ | `simdgroup_matrix<T, 8, 8>` | 32 | 8×8, `SimdgroupMatrix` (B और accumulator), `SimdgroupMatrixT` (A) | 2 |

(यह table उन layouts का set है जिन्हें DSL resolve करता है — `tk/src/arch.rs` में `ArchCaps::frag` और
`tk/src/tiles.rs` के constants। इसके ऊपर हर कर्नेल अपना `ArchSet` ख़ुद declare करता है;
[कर्नेल लाइब्रेरी](./kernel-library) में per-kernel matrix है।)

बस यही एक number है जिसका असर हर चीज़ पर पड़ता है। एक `16×16` tile में 256 elements होते हैं। 64 lanes में
बाँटें तो प्रति lane 4 elements; 32 lanes में बाँटें तो 8 — सिवाय RDNA3 के, जहाँ operand replicated होता है
और हर lane 16 रखती है। अलग-अलग lanes अलग-अलग elements की मालिक होती हैं। तो:

- एक tile का **register layout** बदल जाता है,
- matrix instruction जिस **operand layout** की उम्मीद करता है, वह बदल जाता है,
- और कोई भी **cross-lane reduction** — softmax और layernorm की जान — के steps की संख्या अलग होती है और
  sibling pattern भी अलग।

जो कर्नेल यह hardcode कर देता है कि "64 lanes हैं, lanes 16, 32, 48 को gather करके reduce करो", वह एक
32-lane machine पर एक *अधूरी* reduction compute करता है और चुपचाप ग़लत values लौटा देता है।

---

## हल: shape नहीं, role माँगें

`tk` का जवाब है indirection की एक layer। कोई कर्नेल कभी "16×16, 4 elements per lane" जैसा कोई concrete
fragment shape नहीं लिखता। इसके बजाय यह एक **role** माँगता है, और architecture capabilities को उसे resolve
करने देता है:

```text
   kernel says:  "I need an accumulator fragment"   (FragRole::Accumulator)
                          │
                          ▼
   ArchCaps::frag(role)   ── on CDNA  ──▶  RT_16X16          (wave64, 4/lane)
                          ├─ on gfx11 ──▶  RT_16X16_W32_ACC  (even/odd rows, 8/lane)
                          ├─ on gfx12 ──▶  RT_16X16_GFX12    (strided, 8/lane)
                          ├─ on CUDA  ──▶  RT_16X16_MMA      (two m16n8 halves, 8/lane)
                          └─ on Metal ──▶  RT_8X8_SIMD       (2/lane)
```

roles हैं `FragRole::{Accumulator, Operand, OperandB, AccumulatorT}`, और resolver है `ArchCaps::frag(role)`
(कर्नेल इस तक `ker.frag(role)` से, या shortcuts `ker.acc` / `ker.operand` / `ker.operand_b` / `ker.acc_t`
के ज़रिए पहुँचते हैं)। कर्नेल author बस "accumulator" और "operand" लिखता है; *physical* layout — प्रति lane
element count, interleave map, replication — target के हिसाब से भर दिया जाता है। CDNA और gfx12 हर role को
एक ही shape में resolve करते हैं; gfx11 accumulator, उसके transpose और replicated operand को अलग करता है;
CUDA हर role को two-half map में resolve करता है; Metal A operand को एक उलटा map देता है, क्योंकि उसका core
`D = A·B` सीधे lane map से compute करता है और tk के accumulators column-major हैं। जहाँ tk के पास कोई table
है ही नहीं — pre-Ampere CUDA, pre-Apple7 Metal — वहाँ `None`, ताकि कोई matrix-core कर्नेल ग़लत layout render
करने के बजाय `ker.frag` पर ज़ोर-शोर से fail हो। एक बार लिखो, पाँचों पर चलाओ।

यही सबक़ HipKittens ने भी सीखा था (देखें [tk बनाम HipKittens बनाम CuTile](./comparison)): इसके tile types एक
अकेले compile-time `WARP_THREADS` constant से keyed हैं (CDNA build में `64`), इसलिए अलग wave width का मतलब
है library का एक अलग build। `tk` इस सबको एक ही runtime-resolved `ArchCaps` में समेट देता है।

---

## एक bug जो इसने सचमुच पकड़ा

यह indirection कोई किताबी बात नहीं है। एक शुरुआती `tk` cross-lane all-reduce — वही `shuffle_xor` primitive
जो एक value को पूरी wave भर में sum करने के काम आता है — एक hardcoded wave64 reduction tree के साथ लिखा गया
था। RDNA की 32-lane waves पर इसने उन lanes पर reduce कर दिया जो हिस्सा ही नहीं लेतीं, और ठीक उन्हीं
softmax-style reductions के लिए ग़लत sums निकाल दिए जिन पर attention टिका है। हल था — reduction को किसी
constant से नहीं, बल्कि resolve हुए fragment से चलाना। `tk/src/group/shuffle.rs` में shuffle primitives
`caps.wave_size` पढ़ते हैं; reductions fragment का `LaneMap` पढ़ती हैं; पूरी bug class को design से ही बाहर
कर दिया गया है।

:::tip[GPU विशेषज्ञों के लिए]
दो चीज़ें अधिकांश layout-specific बोझ संभालती हैं, और दोनों resolve हुए fragment के `LaneMap`
(`tk/src/layout.rs`) से पढ़ी जाती हैं, कभी किसी constant से नहीं:

- **Reduce tree।** `LaneMap::tree(wave_size)` in-lane fold के बाद fragment reduce का cross-lane completion है।
  AMD maps हर sibling lane-group का *मूल* partial `ds_bpermute` से gather करते हैं — wave64 पर offsets
  `[16, 32, 48]`, wave32 पर `[16]`; `MmaSync` *running* value को masks `[1, 2]` पर `shfl.bfly` से butterfly
  करता है (एक lane के आठ elements दो rows में फैले होते हैं, इसलिए वह `LaneMap::slots() == 2` values रखती है
  और fold 4-lane quad में पूरा होता है); `SimdgroupMatrix` `[1, 8]` पर butterfly करता है।
  `tk/src/group/reduce.rs` उसे जो भी tree मिले, उस पर चलता है।
- **`acc_reusable_as_input()`** यह जवाब देता है: "क्या एक matrix accumulator को सीधे अगले multiply के
  operand के रूप में वापस feed किया जा सकता है?" CDNA पर true (MFMA accumulator और input `RT_16X16` साझा
  करते हैं), gfx12 पर (हर role के लिए एक fragment), CUDA पर (two-half f32 accumulator `m16n8` C fragments को
  ठीक A-operand वाले register order में रखता है) और Metal पर (B और accumulator एक map साझा करते हैं)। false
  सिर्फ़ gfx11 पर, जहाँ even/odd `<8×f32>` accumulator और replicated `<16×in>` operand अलग होते हैं, इसलिए
  value relayout के लिए LDS से होकर एक round-trip करती है। [Flash Attention](./flash-attention) अपने दो
  matmuls के बीच इसी पर branch करता है।

map एक `LaneArith` trait के ख़िलाफ़ एक बार लिखा जाता है और दो बार evaluate होता है: कर्नेल बनाते वक़्त
`Index`-typed UOps पर, और `tk/src/test/unit/layout.rs` में plain integers पर, जहाँ हर variant को bijection
साबित किया जाता है और Apple map को hardware पर मापी गई एक table से pin किया जाता है। `LaneMap::ldmatrix_x4`
CUDA का `ldmatrix` register plan उसी closed form से निकालता है, उसे हाथ से permute नहीं करता। `BaseShape` पर
`ept` field ([Tiling क्या है](./tiling) से) इसी वजह से मौजूद है: gfx11 पर operands lanes भर में replicate होते
हैं, इसलिए elements-per-thread `element_count / wave_size` नहीं है और इसे explicitly store करना ही पड़ता है।
:::

---

## यह क्यों ज़रूरी है

wave sizes और fragment layouts भर में portability ही वह tax है जो हाथ से लिखे कर्नेल चुकाते हैं, और इसी वजह
से किसी NVIDIA tile library का AMD पर naive port यूँ ही नहीं चल जाता। `tk` यह tax एक बार चुका देता है,
`ArchCaps` और `LaneMap` abstractions में, ताकि अलग-अलग कर्नेल पढ़ने लायक़ बने रहें: वे *roles* में बात करते
हैं और lanes का हिसाब hardware table पर छोड़ देते हैं। और वही abstraction फिर एक कर्नेल को NVIDIA और Apple
*पर* भी ले जाता है — जहाँ warp हमेशा 32 का होता है पर fragment layout core का अपना — यही इसे बनाने का इनाम है।
[Flash Attention](./flash-attention) वह जगह है जहाँ आप इसे एक असली कर्नेल में रंग लाते देखते हैं।
