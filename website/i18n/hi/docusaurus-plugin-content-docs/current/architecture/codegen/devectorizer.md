---
sidebar_label: Devectorizer और index lowering
---

# GPU आयाम, Devectorization और Index Lowering (स्टेज 12–18)

Reduction lowering के बाद भी kernel आकार वाले मानों और `WeakInt` indices की भाषा में ही होता है। ये स्टेज ranges को हार्डवेयर indices पर मैप करते हैं, हर मेमोरी एक्सेस को स्पष्ट scalar `LOAD`/`STORE` बनाते हैं, लगातार (contiguous) एक्सेसों को फिर से चौड़ा करते हैं, और index dtype तय करते हैं। सोर्स: `gpudims.rs`, `devectorize.rs`, `late/coalesce.rs`, `symbolic/index_lowering.rs`।

## 12 — GPU आयाम

दो matchers। `pm_lower_device_ranges` हर renderer के लिए चलता है: `Device` range range की सीमाओं वाला scalar `PARAM` वेरिएबल `_device_num` बन जाता है, और जिस `END` ने उसे बंद किया था वह उस प्रविष्टि को हटा देता है। `pm_add_gpudims` केवल `renderer.has_local || renderer.has_threads` होने पर चलता है; यह `SINK` से एक बार मिलान करता है (`GpuDimsContext` lowered sink id याद रखता है ताकि इंजन की दोबारा विज़िट no-op हो)।

`add_gpudims`:

1. `(axis_id, axis_type)` कुंजी से हर `RANGE` इकट्ठा करता है; यदि कोई `SPECIAL` पहले से मौजूद है तो रुक जाता है।
2. Global dims = `Global` और `Thread` axes; local dims = `Local`, `Warp`, `GroupReduce`। दोनों axis id से क्रमबद्ध; `Warp` axis को locals में सबसे आगे ले जाया जाता है ताकि रैखिक thread index के निचले bits उसके हों (`mma.sync` fragments को हार्डवेयर lane से address करता है)।
3. Index एक्सप्रेशन बनाता है:
   - `has_threads` (CPU): ठीक एक global axis और कोई local नहीं, अन्यथा पास चेतावनी के साथ मना कर देता है। Axis `PARAM("core_id", 0..N-1)` बन जाता है।
   - `KernelInfo.dont_use_locals`: केवल globals, `get_grouped_dims("idx", ..)`।
   - अन्यथा `local_max_axes()` के तहत local shape से `lidx*` (या प्रति axis `local_max`; आगे की सीमा warp extent पर स्थिर रहती है ताकि `lidx0` में कुछ और fold न हो), फिर `global_max` के तहत global shape से `gidx*`, और जब renderer work-item गुणनफल सीमा घोषित करता है तो `global_prod_max / hardware_local_extents` से और सीमित।
4. `get_grouped_dims` Tinygrad वाला है: यदि dims प्रति-axis सीमाओं में नहीं समाते, तो `group_dims` उन पड़ोसी dims को मिलाता है जिनका गुणनफल समा जाए; यदि कुछ भी समूहित न हो सके, तो `split_dims` बहुत बड़े dim को उसके सबसे छोटे भाजक से अगले स्लॉट में बाँटता है। कोई भी विफलता codegen के बजाय scheduling के समय panic करती है (`"cannot limit dims to N axes"`)। परिणाम हर सीमित dim के लिए एक `SPECIAL(end, "gidxN")` है; जब समूहन या विभाजन हुआ हो, तो हर मूल dim को समतल index से `FloorDiv`/`FloorMod` द्वारा फिर से बनाया जाता है और `symbolic` से सरल किया जाता है। Global indices `reverse = true` के साथ बनते हैं (recursion input *और* output दोनों को उलटता है, इसलिए नाम iteration क्रम में रहते हैं)।
5. **Store masking** (`compute_store_masks`): global मेमोरी में ऐसा `STORE` जिसका index हर local range के scope में नहीं है, अपने index पर `WHERE((l1 == 0) & (l2 == 0) & .., idx, Invalid)` पाता है, ताकि हर अप्रयुक्त local axis के लिए केवल एक work-item लिखे। Mask index एक्सप्रेशन के अंदर रहता है ताकि RANGE → SPECIAL प्रतिस्थापन उसे हार्डवेयर index तक ले जाए।
6. हर GPU range को उसके index से बदलता है; `Reduce` ranges लूप बने रहते हैं।

[उदाहरण सहित विवरण](./worked-example.md) पेज के CPU उदाहरण में कुछ नहीं होता: row axis `Weak` है (कोई `Thread` axis नहीं बना) और renderer में locals नहीं हैं।

## 13 — loads

`PM_ADD_LOADS = symbolic_simple() + pm_expand_broadcast() + pm_add_loads()`।

- `pm_expand_broadcast` फिर से `pm_wmma_add` से शुरू होता है, फिर broadcasting को स्पष्ट बनाता है: ऐसा `Binary`/`Ternary`/`STORE` जिसके sources के shapes अलग हैं, हर source को `RESHAPE` (आगे 1s) और broadcast shape तक `EXPAND` करवाता है; ऐसा `WMMA` जिसके operand prefixes अलग हैं, हर output coordinate के लिए फैलाया जाता है (`broadcast_and_devec_wmma`)।
- `pm_add_loads` हर उस operand को `LOAD` में लपेटता है जो *मान के रूप में उपभोग* होता है: ALU ops, casts, `REDUCE`, `WMMA` और `STACK` के वे sources जिनका address space है (`maybe_load`), और ऐसा `STORE` मान जो स्वयं एक address है। Address के रूप में प्रयुक्त `INDEX` — `STORE` लक्ष्य, `WMMA` fragment pointer — बिना लपेटे रहता है। स्टेज 10 पर बने accumulator reads (`AFTER(acc, ..)`) यहाँ `LOAD(AFTER(acc, ..))` बन जाते हैं।

## 14 — devectorize

`devectorize()` `symbolic_simple + devectorize_patterns + bool_storage_patterns + indexing_simplify` पर एक `graph_rewrite` है (`Renderer` context, जिसे नियम उपयोग नहीं करते)। कोई बाहरी लूप नहीं है: इंजन हर प्रतिस्थापन का फिर से मिलान करता है।

`devectorize_patterns` (Tinygrad में `devectorizer2`), सोर्स क्रम में:

| समूह | नियम |
|-------|-------|
| `movement_cleanup_patterns` | `mop_cleanup_patterns` के साथ `RESHAPE(STACK([x]))` और आगे के singleton materializations |
| `movement_op_patterns` | rangeify के movement नियम (`INDEX`, `AFTER`, `END` के पार) |
| `no_vectorized_alu` | ख़ाली न होने वाले shape के साथ हर unary/binary/ternary op, `CAST`, `BITCAST` → `devectorize_alu` |
| `mixed_representation_alu` | ऐसा ALU जिसके sources में `STACK` और vector-dtype मान मिले हों: vector sources को `STACK(INDEX(src, lane)..)` में खोला जाता है, फिर `devectorize_alu` |
| आकार वाले `LOAD` / `STORE` | → `devectorize_alu` (प्रति-lane `LOAD(INDEX)`; प्रति-lane stores एक `GROUP` में इकट्ठे) |
| `INDEX(buf, [])` | → `buf` |
| `WMMA` | `stack_wmma_sources`: operands load हुए lanes के `STACK`s बन जाते हैं |
| `PARAM`/`BUFFER` पर `INDEX(buf, STACK(i0, i1, ..))` | → `STACK(INDEX(buf, i0), INDEX(buf, i1), ..)` — lanes addresses बने रहते हैं; घेरने वाला `LOAD`/`STORE` उन्हें materialize करता है |
| `INDEX(buf, RESHAPE(i))` | → `RESHAPE(INDEX(buf, i))` |
| `Void` मान का `RESHAPE` | → वही मान (`AFTER`/`STORE` के आसपास shape का हिसाब) |
| scalar में reshape किया गया एक-तत्व वाला आकार वाला मान | → `INDEX(src, 0)` |
| `EXPAND` | `materialize_stack_broadcast` (N lanes तक broadcast किया गया `STACK([x])` → `STACK([x; N])`) या `expand_scalar_to_stack` |

`devectorize_alu` Tinygrad का `do_devectorize` है: यह माँगता है कि हर source का shape परिणाम जैसा हो (या वह `Invalid` base हो, जिसका scalar बहुरूपी है), स्थिर shape के coordinates गिनता है, `INDEX(source, c0, c1, ..)` operands के साथ हर coordinate के लिए एक scalar op बनाता है, और `stack_with_shape` से फिर जोड़ता है (shape को दर्शाने वाले नेस्टेड `STACK`s) — या `STORE` के लिए `GROUP`। Lane संख्या shape का पूरा गुणनफल है; कोई प्रति-डिवाइस fold चौड़ाई नहीं है। फिर से vectorize करना backend का काम है (LLVM का SLP vectorizer, या मेमोरी के लिए दो स्टेज बाद `memory_coalescing`)।

`bool_storage_patterns`: bool `STORE` `uint8` में cast होता है, bool `LOAD` `uint8` load करके वापस cast करता है, bool को छूने वाला `BITCAST` `CAST` बन जाता है। LLVM का `i1` ऊपरी bits में कचरा रख सकता है।

`indexing_simplify` (`late/coalesce.rs`): `INDEX(buf, WHERE(valid, idx, Invalid))` के लिए, `uop_given_valid` यह मानकर `idx` को फिर से लिखता है कि `valid` सत्य है (`symbolic/valid_simplification.rs`); दो-coordinate वाला image रूप अतिरिक्त रूप से उन validity clauses को हटाता है जो image की सीमाओं से पहले ही निहित हैं (`drop_valid_stmts`)।

इस स्टेज के बाद हर ALU op scalar है। उदाहरण में index एक्सप्रेशन के चार lanes चार `LOAD(INDEX(PARAM, R0*4 + R1*64 + k))` बन जाते हैं और क्षैतिज `Add` शृंखला स्पष्ट हो जाती है।

## 15 — early symbolic

एक बार फिर `sym()`, अब scalar कोड पर। इसके होने का कारण अगला स्टेज है: coalescing उन्हें समूहित कर सके, इसके लिए index एक्सप्रेशन canonical `base + const` रूप में होने चाहिए।

## 16 — memory coalescing

`memory_coalescing` (`late/coalesce.rs`) एक ग्राफ़-वॉक है, matcher नहीं। यह बिना gate वाले `LOAD`s और `STORE`s को `(op, buffer, index base, validity)` से समूहित करता है, जहाँ index को `base + integer_offset` में बाँटा जाता है (`Invalid` या constant index स्वयं अपना base है)। एक समूह के भीतर, लगातार offsets runs बनाते हैं; हर run को सबसे चौड़ी उस fold लंबाई में काटा जाता है जो base offset को विभाजित करती है:

- image buffers: 4;
- `supports_float4` renderers: `access_bytes() >= 16` होने पर `16 / sizeof(dtype)` से नीचे की ओर दो की घातें (आठ 16-bit lanes, चार `f32`), अन्यथा 4 से;
- अन्यथा केवल scalar। `Reg` buffers और fold न होने वाले dtypes (f32/f16/bf16/i32/u32/fp8 के अलावा कुछ भी) scalar रहते हैं।

चौड़ाई `n > 1` वाला fold `LOAD(SHRINK(buf, offset, n))` बन जाता है, जिसमें पुराने loads `INDEX(load, lane)` से बदले जाते हैं, या `STORE(SHRINK(..), STACK(values))`। `SHRINK` समूह का shape रखता है; मेमोरी dtype scalar रहता है। `DMC=1` पास को बंद करता है। उदाहरण में चार unrolled loads एक `LOAD(SHRINK(PARAM(1), R0*4 + R1*64, 4))` बन जाते हैं, जिसे LLVM backend `load <4 x float>` के रूप में render करता है।

## 17 — bottom-up elementwise / image पास

`symbolic_simple + no_vectorized_alu + pm_simplify_add_image`, `graph_rewrite_bottom_up` और एक `AddImageContext` के साथ लागू। Image नियम f32 image buffers पर f16 एक्सेस को canonical बनाते हैं (`LOAD` → `LOAD.cast(f16)`, `STORE(value.cast(f32))`, `CAST(CAST(x, f16), f32)` round-trip हटाना)। Image buffer *निर्माण* का कोई Svod target नहीं है; नियम केवल मौजूदा image एक्सेस की सेवा करते हैं। `no_vectorized_alu` फिर से चलता है क्योंकि image rewrite आकार वाला op फिर से ला सकता है।

## 16 — extra symbolic

`extra_symbolic_patterns = sym() + indexing_simplify()`। यहाँ indices जानबूझकर अभी भी `WeakInt` हैं: `sym` और `indexing_simplify` के distributive और index-validity नियमों को weak dtype चाहिए, इसलिए यह उनका आख़िरी मौका है। (लेबल memory coalescing वाले से टकराता है; दोनों `SVOD_DUMP_STAGE=16` के तहत प्रिंट होते हैं।)

## 17 — index dtype lowering

`lower_index_patterns = symbolic_simple + pm_fold_cast_const + pm_lower_index_dtype + indexing_simplify`, प्रति kernel एक `WeakMemo` के साथ (Tinygrad का एकल `ctx={}`)। `tinygrad/uop/weak.py` का port।

`select_dtype(u)`: `WeakFloat` → डिफ़ॉल्ट float; ऐसा integer जिसके `vmin`/`vmax` `i32` में समाते हों → डिफ़ॉल्ट int, अन्यथा `Int64`; vector count बना रहता है।

`pm_lower_index_dtype` इन्हें जोड़ता है:

1. `pm_commit_weak` — weak source और non-weak `least_upper_dtype` वाला `Binary`/`Ternary` weak sources को उस पर commit करता है (`commit_weak`: `CONST` का type बदला जाता है, बाकी सब cast होता है); ऐसा `STORE` जिसका मान weak है, उसे index के dtype पर commit करता है।
2. `pm_cast_weak` — `CAST(weak_alu, concrete)` concrete dtype को ALU के sources में धकेलता है।
3. `SHRINK` offsets/sizes `select_dtype` से commit किए जाते हैं।
4. weak sources वाला कोई भी non-weak नोड → `lower_weak_srcs`: हर weak source को `pm_lower_weak` से फिर से लिखा जाता है (source id से memoized) और पीछे का weak `CAST` उपभोक्ता के अपने edge में समा जाता है। `pm_lower_weak` तीन-चरणीय cascade है:
   - leaves: `CONST`/`VCONST`/scalar `PARAM` `concrete.cast(weak)` बन जाते हैं;
   - `Unary`, `Binary`, `WHERE` (शर्त छोड़ी जाती है), `RANGE`, `STACK`, `SPECIAL` (`lower_weak_node`): sources पर weak casts खोलो, concrete dtype निकालो (binary ops के लिए `select_dtype(u)` और sources का `least_upper_dtype`; अन्यथा `dtype_from_op`), हर source को उसमें cast करो, परिणाम पर weak `CAST` रखो जब तक वह `STACK` न हो;
   - weak dtype वाला `INDEX`: buffer को चुने गए dtype में cast किया जाता है और हर weak index commit किया जाता है;
   - `CAST(weak, CAST(weak, x))`: अंदर का cast commit होता है, बाहर वाला रखा जाता है।
5. ऐसा `INDEX` (या `SHRINK`) जिसका gated index पहले ही `Int64` निकला हो, buffer की तत्व संख्या `i32` में समाने पर वापस `Int32` में संकुचित किया जाता है।

यही वह स्टेज है जहाँ `WHERE(valid, idx, Invalid)` अपना आकार बनाए रखता है: `lower_weak_node` `Invalid` sources को नहीं छूता। उदाहरण में हर `WeakInt` `Int32` बन जाता है (`RANGE(R0, Reduce) : Scalar(Int32)`), और `PARAM` sizes `Int32` constants बन जाते हैं।

## 18 — final symbolic

ठोस type वाले ग्राफ़ पर `symbolic()` (tier 2, बिना `pm_simplify_valid`/lane folds)। `SVOD_SPEC` चालू होने पर, `verify_no_legacy_index_dtype` assert करता है कि कोई `WeakInt` नहीं बचा।
