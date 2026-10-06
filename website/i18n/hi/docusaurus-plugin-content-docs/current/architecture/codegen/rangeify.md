---
sidebar_label: Rangeify और कर्नेल कट
---

# Rangeify, कर्नेल कट और प्री-ऑप्टिमाइज़ेशन

इस पेज पर सब कुछ ऑप्टिमाइज़र के कर्नेल देखने से पहले चलता है। स्रोत: `schedule/src/rangeify/` और `schedule/src/optimizer/mod.rs` में `apply_pre_optimization`।

## Rangeify (`rangeify_with_map`)

इनपुट: वह टेंसर ग्राफ़ जो `realize()` कॉल ने बनाया (मूवमेंट ops, टेंसर रूप में `num_axes > 0` वाला `REDUCE`, `CONTIGUOUS`, `COPY`, ...)। आउटपुट: ऐसा ग्राफ़ जिसमें हर लूप एक स्पष्ट `RANGE` है, हर मटीरियलाइज़ेशन एक `STAGE` है, और हर रीड एक `INDEX` है।

पास, क्रम से (`rangeify/transforms.rs`):

1. **मल्टी-डिवाइस समाधान** — `multi_pm` फिर `lower_allreduce_pm` (दोनों `graph_rewrite_preserve_calls`); `validate_supported_subset` उसे अस्वीकार करता है जिसे बैकएंड नहीं चला सकते।
2. **`add_tags_patterns`** (bottom-up) हर टैग-योग्य नोड को `[i]` नंबर देता है। टैग टेंसर की पहचान हैं: कट के बाद आउटपुट मैप बचे हुए टैग से फिर से बनाया जाता है। `PARAM`, `CONST`, `RANGE`, `END`, `CALL`, मूवमेंट ops और सभी-PARAM वाले `MSTACK`/`MSELECT` टैग नहीं किए जाते।
3. **`resolve_calls`** `FUNCTION` बॉडी में उनके आर्ग्युमेंट प्रतिस्थापित करता है और `GETTUPLE(TUPLE(..), i)` को फ़ोल्ड करता है। प्रीकंपाइल्ड फ़ंक्शन और `CALL` आर्ग्युमेंट अपारदर्शी रहते हैं।
4. **सबसे शुरुआती रीराइट** (bottom-up, एक मैचर): `movement_op_patterns + early_rewrites + split_reduceop_patterns`। `early_rewrites` `DETACH`/`CONTIGUOUS_BACKWARD` हटाता है, बिना टैग वाली `RESHAPE` शृंखलाओं को मिलाता है, widening cast के तहत पूर्णांक गुणनफलों को चौड़ा करता है, आकार बदले/पुनर्क्रमित `COPY` स्रोत को `CONTIGUOUS` से मटीरियलाइज़ करता है, समान-डिवाइस `COPY` हटाता है, और शून्य-आकार टेंसर को स्थिरांकों में फ़ोल्ड करता है। `split_reduceop` दो-चरणीय रिडक्शन विभाजन है (देखें [रेंज ऑप्टिमाइज़ेशन](../optimizations/range-optimization.md))।
5. **`run_rangeify`** (`rangeify/indexing.rs`):
   - `pm_generate_realize_map` (bottom-up): चिह्नित करता है कि क्या बफ़र बनना चाहिए — `STORE`, `CONTIGUOUS`, `COPY` और उनके non-contiguous स्रोत, `MSTACK`/`MSELECT` स्रोत, और हाथ से लिखे कर्नेल `CALL` के इनपुट (पिन किए गए, हटाए नहीं जा सकते)।
   - `assign_ranges`: रूट से पत्ती तक की यात्रा। रियलाइज़्ड नोड को हर आउटपुट डाइमेंशन के लिए नई `Weak` रेंज मिलती हैं (`IndexingContext::new_range`; आकार-1 डाइमेंशन `CONST(0)` है)। बाकी नोड अपने उपभोक्ताओं की रेंज विरासत में लेते हैं; जब उपभोक्ता असहमत हों, तो `merge_consumer_ranges` या तो संगत इंडेक्स एक्सप्रेशन मिलाता है (वैध भाग `WHERE(valid, idx, Invalid)` में OR किए जाते हैं) या नई रेंज आवंटित करके अक्ष को रियलाइज़ेशन के लिए चिह्नित करता है। मूवमेंट ops `apply_movement_op` से आउटपुट रेंज को इनपुट रेंज पर मैप करते हैं (`PERMUTE` उन्हें क्रमचयित करता है, `EXPAND` ब्रॉडकास्ट अक्ष को शून्य करता है, `PAD` रेंज को वैधता `WHERE` में लपेटता है, `RESHAPE` `apply_reshape_ranges` से गुज़रता है)। `ending_ranges` ब्रॉडकास्ट निर्णयों को पीछे की ओर फैलाते हैं ताकि ब्रॉडकास्ट को पोषित करने वाला `REDUCE` उससे पहले रियलाइज़ हो (layernorm का मामला)।
   - `apply_rangeify_patterns` (bottom-up): टेंसर-रूप `REDUCE` → `num_axes = 0` वाला लूप-रूप `REDUCE(src, ranges)`; `PAD` → `WHERE(valid, src, 0)`; आकार वाला `STACK` → उसकी अग्रणी रेंज पर `WHERE` शृंखला; हर op के रियलाइज़्ड स्रोत `STAGE` + `INDEX` में लपेटे जाते हैं (`transform_sources_with_bufferize`); फिर मूवमेंट ops हटा दिए जाते हैं। बफ़र-जैसे स्रोत (`BUFFER`, `PARAM`, `SLICE`, `AFTER`, ...) को स्थिर आकार होने पर एक row-major `INDEX` मिलता है (`linearize_static_indices`); इमेज और सिम्बॉलिक आकार प्रति निर्देशांक एक इंडेक्स रखते हैं।
6. **मेगा-पास** — `symbolic + pm_reduce_simplify + movement_op_patterns + buffer_folding + dead_axis_removal + pm_remove_bufferize` पर एक फ़िक्सपॉइंट। समूह एक-दूसरे को पोषित करते हैं: `STAGE` को इनलाइन करने से रेंज अंकगणित उजागर होता है जिसे `symbolic` फ़ोल्ड करता है, जिससे reduce संकुचित होने योग्य बन सकता है। अलग-अलग नियम [रेंज ऑप्टिमाइज़ेशन](../optimizations/range-optimization.md) पेज पर हैं।
7. **SINK पुनर्निर्माण** टैग किए गए backward slice से: सिर्फ़ आउटपुट टैग वाले `STAGE`, `MSTACK`, `CONST`, `PARAM` और `AFTER` नोड sink स्रोत के रूप में रहते हैं, मूल आउटपुट क्रम में।
8. **बफ़र सीमा** — यदि डिवाइस `max_buffers` बताता है, तो `buffer_limit_patterns` elementwise स्रोतों को global `STAGE` में बाध्य करता है ताकि कोई कर्नेल आर्ग्युमेंट सीमा से अधिक न हो।

`[8, 64]` टेंसर पर `x.sum(1)` का परिणाम:

```text
[67] SINK : Scalar(Void)
└── [66] STAGE : Scalar(Float32) shape=[Const(8)]
    ├── [65] CONTIGUOUS : Scalar(Float32) shape=[]
    │   └── [64] REDUCE(Add, num_axes=0, ranges=[27]) : Scalar(Float32) shape=[]
    │       ├── [62] INDEX : Scalar(Float32) shape=[]
    │       │   ├── [11] PARAM(slot=0) : Scalar(Float32) shape=[Const(512)]
    │       │   │   └── [0] CONST(Int(512)) : Scalar(WeakInt) shape=[]
    │       │   └── [55] Add : Scalar(WeakInt) shape=[]
    │       │       ├── [54] Mul : Scalar(WeakInt) shape=[]
    │       │       │   ├── [26] RANGE(U0, Weak) : Scalar(WeakInt) shape=[]
    │       │       │   │   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
    │       │       │   └── [3] CONST(Int(64)) : Scalar(WeakInt) shape=[]
    │       │       └── [27] RANGE(U1, Reduce) : Scalar(WeakInt) shape=[]
    │       │           └── [3] → (see above)
    │       └── [27] → (see above)
    └── [26] → (see above)
```

`U0`/`U1` `AxisId::Unrenumbered` हैं: कट पर रेंज प्रति कर्नेल फिर से नंबर की जाती हैं। इनपुट का `PERMUTE`/`RESHAPE` गायब है — वे इंडेक्स एक्सप्रेशन `U0 * 64 + U1` बन गए।

## कर्नेल कट (`try_get_kernel_graph`)

पहले `kernel_graph_pre_cut`:

- **`pm_add_buffers_patterns`** (bottom-up, `RangeifyBufferContext`): `movement_op_patterns`, फिर `flatten_bufferize` (बहु-रेंज `STAGE` एक फ़्लैट रेंज और वापस `RESHAPE` बन जाता है), `late_buffer_slice` (DISK `STAGE(BITCAST|CONTIGUOUS)` एक `SLICE` बन जाता है), और `bufferize_to_store`। आख़िरी वाला एक शेड्यूल-लोकल `BUFFER` आवंटित करता है (`new_lunique_buffer`, high-bit नेमस्पेस में स्लॉट) और `STAGE(compute, ranges)` को `AFTER(BUFFER, [END(STORE(INDEX(BUFFER, idx), compute), ranges)])` में बदलता है। `STAGE(AFTER(..))` अंतर्निहित बफ़र का पुन: उपयोग करता है; `Local` `STAGE` को बाद में `pm_add_local_buffers` के लिए छोड़ दिया जाता है। पहले से बना कर्नेल `SINK` (जिसमें `KernelInfo` हो) गेट किया जाता है ताकि रीराइट उसमें न उतरे।
- **`pm_flatten_range`** पूरे ग्राफ़ पर एक बार (bottom-up): हर `END`/`REDUCE` की रेंज सूची को उसके स्रोतों से पहुँचने योग्य `RANGE` से फिर से निकालता है, ताकि नीचे का प्रति-कर्नेल पास साझा सबग्राफ़ को फिर से न खंगाले।

फिर **`split_all_stores`** (bottom-up): हर `STORE` या `END(STORE)` जिसकी कोई कम्प्यूटेशनल रेंज खुली न हो, एक `CALL` बन जाता है। `split_store` कर्नेल बॉडी पर `local_to_param_patterns + rangeify_codegen_patterns` चलाता है: global `BUFFER`/`PARAM` → codegen `PARAM(slot)`, जिसे `LocalAddBufferContext::param_slot` मैच क्रम में नंबर देता है; `BIND(var, value)` → वेरिएबल, बाइंडिंग `CALL` आर्ग्युमेंट के रूप में रखी जाती है; `AFTER`/`MSTACK`/`MSELECT` → उनका बफ़र; `RANGE(end=0)` → `CONST(0)`; `Unrenumbered` अक्ष ids → `Renumbered(n)`; `NOOP` → टाइप वाला शून्य; `CONTIGUOUS` → उसका स्रोत, hints एकत्रित। बॉडी को डिफ़ॉल्ट `KernelInfo` के साथ `SINK` में लपेटा जाता है; `COPY`/`SLICE` मान सीधे call बॉडी रहता है। `Device` रेंज "कोई खुली रेंज नहीं" का एकमात्र अपवाद हैं: वे लॉन्च लेन हैं और सीमा के पार बची रहती हैं।

अंत में **`validate_normal_kernel_devices`** (हर non-copy कर्नेल के लिए एक डिवाइस) और **`fix_assign`**: जब कर्नेल B वह बफ़र पढ़ता है जिसे कर्नेल A लिखता है, तो A का `AFTER` B के `AFTER` deps में जोड़ा जाता है; चक्र `KernelSplitDependencyCycle` है। `SVOD_SPEC` चालू होने पर `verify_kernel_graph` परिणाम जाँचता है।

## प्रति-कर्नेल प्री-ऑप्टिमाइज़ेशन (`apply_pre_optimization`)

ह्यूरिस्टिक्स या BEAM से पहले हर कर्नेल बॉडी पर चलता है, दोनों पथों में (`optimize_kernel_with_config_impl`, `optimize_kernel_beam`, `prepare_scheduler`)। `SVOD_SPEC` चालू होने पर पहले `spec_tensor` के विरुद्ध `type_verify` चलता है।

| चरण | मैचर | दिशा |
|------|---------|-----------|
| मूवमेंट ops | `movement_op_patterns` | bottom-up |
| load collapse | `pm_load_collapse` | top-down |
| रेंज विभाजन | `pm_split_ranges + pm_flatten_range` (`SplitRangesContext`) | top-down |
| symbolic | `sym + pm_fold_cast_const + pm_flatten_range` | top-down |
| रेंज सरलीकरण | `pm_flatten_range + pm_simplify_ranges` (`SimplifyRangesContext`) | top-down |

**`movement_op_patterns`** में तीन नियम हैं: `INDEX(mop(x), idx)` → `INDEX(x, mop⁻¹(idx))` (`transform_movement_through_index`), `AFTER(mop(x) | INDEX(x), deps)` → `mop(AFTER(x, deps))` (`push_op_through_after`), और `END(mop(x), ranges)` → `END(x, ranges)`। `is_movement()` ठीक-ठीक `RESHAPE`, `PERMUTE`, `EXPAND`, `PAD`, `SHRINK`, `FLIP` है। इसे bottom-up लागू किया जाता है क्योंकि भीतरी मूवमेंट op को उसके उपभोक्ता के मैच होने से पहले फिर से लिखा जाना चाहिए।

**`pm_load_collapse`** ऐसा `REDUCE(Add)` हटाता है जिसकी बॉडी सिम्बॉलिक तर्क के बाद रेंज-स्वतंत्र हो (`reduce_load_collapse`): reduce दायरे के बाहर के नोड स्केलर `PARAM` वेरिएबल (`UOp::variable("in{n}", vmin, vmax)`) से बदले जाते हैं, बॉडी को एक रेंज पर एक कृत्रिम `REDUCE` में लपेटा जाता है, `build_reduce_load_collapse_matcher` चलता है, और यदि कोई `RANGE` बचता नहीं तो प्रतिस्थापन उलट दिया जाता है। इसके द्वारा उपयोग किए जाने वाले बाउंड पैटर्न [रेंज ऑप्टिमाइज़ेशन](../optimizations/range-optimization.md) पेज पर हैं।

**`pm_split_ranges`** हर `RANGE % const` दर्ज करता है जिसका end स्थिरांक से विभाज्य हो (`Warp` और `Device` रेंज को छोड़कर; इमेज `STORE` द्वारा इंडेक्स की गई हर रेंज पिन होती है) और `SINK` पर एक बार `r → outer * c + inner` प्रतिस्थापित करता है, अक्ष ids `r.child(0)` / `r.child(1)` के साथ। प्रतिस्थापित ग्राफ़ फिर `symbolic + pm_fold_cast_const` से सरल किया जाता है।

**`sym`** पूरा tier-3 सरलीकारक है ([बीजगणितीय सरलीकरण](../optimizations/algebraic-simplification.md)); `pm_fold_cast_const` `CAST(CONST)` फ़ोल्ड करता है; `pm_flatten_range` रेंज गायब होने के बाद रेंज सूचियों को सही रखता है।

**`pm_simplify_ranges`** `END`/`REDUCE` की आसन्न रेंज को मिलाता है जब मिला हुआ रूप `FloorDiv`/`FloorMod` की गिनती नहीं बढ़ाता (`simplify_merge_adjacent`), और रेंज को उस सबसे बड़े बाउंड तक संकुचित करता है जिसे कोई `INDEX` गेट उसके लिए सिद्ध करे (`mark_gated`; एक भी बिना गेट उपयोग मूल end को पिन कर देता है; `REDUCE` रेंज सुरक्षित हैं)। दोनों प्रतिस्थापन `SINK` पर होते हैं।

## ऑप्टिमाइज़र को सौंपना

`Scheduler::new(ast, renderer)` extent > 1 वाली `RANGE` एकत्र करता है, `(axis_type.priority(), axis_id)` के क्रम में; `convert_loop_to_global` `has_local` वाले रेंडरर पर `Weak` आउटपुट अक्षों को `Global` में बदलता है (CPU पर यह no-op है, इसीलिए ऊपर के उदाहरण में row अक्ष `Weak` रहता है)। फिर `hand_coded_optimizations` या BEAM `Opt` लागू करता है और `get_optimized_ast_with_naming` `KernelInfo` मेटाडेटा (नाम जैसे `r_8_16_4`, `dont_use_locals`, `opts_to_apply`) के साथ कर्नेल `SINK` बनाता है। `SVOD_NOOPT` ह्यूरिस्टिक्स छोड़ देता है, लेकिन न यह पेज और न ही post-optimization चरण। सर्च स्वयं [कर्नेल सर्च](../optimizations/kernel-search.md) में वर्णित है।
