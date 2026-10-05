---
sidebar_label: Expander और reductions
---

# Expander और Reduction Lowering (स्टेज 08–11)

पहले चार post-optimization स्टेज ऑप्टिमाइज़र का आउटपुट लेते हैं — एक kernel जिसके `RANGE`s अब `Upcast`/`Unroll`/`Global`/`Local`/`GroupReduce` axis types रखते हैं — और उस इरादे को ठोस बनाते हैं: unrolled ranges आकार वाले constants बन जाते हैं, `REDUCE` एक accumulator लूप बन जाता है, local `STAGE`s local buffers बन जाते हैं। ये सभी `apply_post_optimization_configured_with_capture` (`optimizer/mod.rs`) के अंदर चलते हैं।

## 08 — post-opt symbolic

`POST_OPT_SYM = sym() + pm_move_where_on_load() + pm_flatten_range() + pm_reduce_unparented()`, एक top-down fixpoint। सोर्स में क्रम मायने रखता है: बाद वाले समूह वह उपभोग करते हैं जो पहले वाले बनाते हैं।

- `sym()` पूरा tier-3 simplifier है ([बीजगणितीय सरलीकरण](../optimizations/algebraic-simplification.md))।
- `pm_move_where_on_load` (`symbolic/patterns.rs`) `WHERE(cond, INDEX(buf, idx), 0)` को `INDEX(buf, WHERE(cond', idx, Invalid))` में बदलता है। शर्त को `AND` पर विभाजित किया जाता है; कोई clause index में तभी जाता है जब उसके सभी ranges `INDEX` के scope में हों और उसकी अपनी कोई `INDEX` निर्भरता न हो; बाकी clauses एक बाहरी `WHERE` में रहते हैं। उल्टे रूप `WHERE(cond, 0, INDEX(..))` को negated शर्त के साथ संभाला जाता है। Validity अब index एक्सप्रेशन के अंदर चलती है, जहाँ devectorizer और `indexing_simplify` उसे देख सकते हैं; यह LOAD/STORE `gate` केवल `19e` पर बनती है।
- `pm_flatten_range` `END`/`REDUCE` की range सूचियाँ फिर से बनाता है।
- `pm_reduce_unparented` उन reduce ranges को हटाता है जिन्हें body संदर्भित नहीं करती: `Add` extent से गुणा करता है, `Mul` extent की घात लेता है, `Max` बस range हटा देता है (कोई `Min` शाखा नहीं है; `Min` reductions का मिलान नहीं होता)।

## 09 — expander (`pre_expand`)

`RangeMap` context (`expand.rs`) के साथ `expander2() + pm_flatten_range() + mop_cleanup_patterns()`। `build_range_map` हर `Upcast`/`Unroll` `RANGE` को toposort क्रम में एक coordinate स्थान देता है; map की लंबाई उन आकार वाले मानों की rank है जो यह स्टेज बनाता है।

तीन नियम, सोर्स क्रम में:

| नियम | प्रभाव |
|------|--------|
| `Reduce { .. }` → `expand_reduce` | लूप-रूप वाला `REDUCE` जिसकी range सूची में आकार वाली non-`RANGE` प्रविष्टियाँ हों, उन प्रविष्टियों के axes (extent > 1) को आगे के *क्षैतिज* axes में बदल देता है: source को permute किया जाता है ताकि वे पहले आएँ और `num_axes` उन्हें गिनता है; परिणाम को size-1 placeholders बनाए रखने के लिए reshape किया जाता है। |
| `Range { axis_type: Upcast \| Unroll }` → `expand_range` | Range `RESHAPE(STACK(CONST(0), ..., CONST(end-1)), shape)` बन जाता है, जहाँ `shape` में range के अपने coordinate को छोड़कर सब 1 हैं। Range का हर उपभोक्ता broadcasting से आकार वाला बन जाता है; अभी कुछ भी दोहराया नहीं जाता। |
| `Wmma { metadata.upcast_axes: Some(..) }` → `expand_wmma` | `contract_axis` A/B upcast coordinates को अंत में ले जाता है और उन्हें fragment operands में समतल करता है; `unroll_axis` आउटपुट पर C coordinates वापस लाता है। Metadata का `upcast_axes` साफ़ कर दिया जाता है। |

`mop_cleanup_patterns` (`devectorize.rs`) Tinygrad का `mop_cleanup` है: नेस्टेड `RESHAPE`s को मिलाना, identity `RESHAPE`/`PERMUTE` हटाना, `PERMUTE` शृंखलाओं को मिलाना, `STACK(INDEX(b,0), INDEX(b,1), ..)` को वापस `b` में समेटना, `INDEX(STACK(..), const)` को lane में fold करना, और indices scalar होने पर `INDEX(INDEX(b, i), j)` को `INDEX(b, i, j)` में जोड़ना। यहाँ कोई symbolic matcher नहीं चलता।

उदाहरण में reduce range `R2` (`Unroll`, extent 4) गायब हो जाता है और index आकार वाला बन जाता है:

```text
[151] REDUCE(Add, num_axes=1, ranges=[118]) : Scalar(Float32) shape=[]
├── [149] INDEX : Scalar(Float32) shape=[Const(4)]
│   ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
│   └── [148] Add : Scalar(WeakInt) shape=[Const(4)]
│       ├── [147] Add : Scalar(WeakInt) shape=[Const(4)]
│       │   ├── [119] Mul : Scalar(WeakInt) shape=[]          ← R0 * 4
│       │   └── [146] STACK(len=4) : Scalar(WeakInt) shape=[Const(4)]
│       └── [90] Mul : Scalar(WeakInt) shape=[]              ← R1 * 64
└── [118] RANGE(R0, Reduce)
```

`expand_reduce` पहले ही 4-चौड़े lane axis को `num_axes=1` में बदल चुका है, इसलिए lanes पर reduction क्षैतिज है और बचा हुआ लूप केवल `R0` पर है।

:::tip[STACK ही एकमात्र vector op है]
आकार वाला मान lanes का एक `STACK` होता है (संभवतः नेस्टेड, संभवतः किसी `RESHAPE` के पीछे)। `INDEX(STACK(..), c)` उसी op से lane चुनता है जो buffer को address करता है। कोई vectorize/contract op जोड़ी नहीं है, और `Upcast`/`Unroll` `AxisType`s हैं, ops नहीं।
:::

## 10 — reduction lowering (`pm_reduce`)

`ReduceContext` के साथ `movement_cleanup_patterns() + pm_reduce_local()`। `movement_cleanup_patterns` में `mop_cleanup_patterns` के साथ दो केवल-devectorizer नियम हैं (shapes मेल खाने पर `RESHAPE(STACK([x]))` → `x`; केवल आगे के 1-dims जोड़ने वाला `RESHAPE` → हर जोड़े गए dim के लिए एक `STACK([..])` wrapper)।

`pm_reduce_local` (`devectorize.rs`) क्रम से इन्हें जोड़ता है:

1. **`pm_wmma_add`** — `WMMA(a, b, c) + add` → `WMMA(a, b, c + add)`, उस `PERMUTE` और `PERMUTE(RESHAPE(..))` wrapper के पार भी जो `expand_wmma` ने आउटपुट पर छोड़ा था। dtype मेल न खाने पर `try_add` assert करने के बजाय मना कर देता है।
2. **`pm_group_for_reduce`** (`expand.rs`) — `GroupReduce` ranges वाला `REDUCE` यह बन जाता है: अन्य ranges पर आंशिक `REDUCE` → scope में मौजूद `Local` ranges और group ranges के साथ आंशिक का `STAGE` (`BufferizeOpts::local_for_axis`) → locals और नए `Reduce` लूपों (`axis_id.group_reduce_loop()`) के साथ उस stage का `INDEX` → उन लूपों पर अंतिम `REDUCE`।
3. **`reduce_to_acc`** — ranges वाला `REDUCE`। यदि `num_axes > 0` है तो lanes को पहले row-major क्रम में बाएँ से दाएँ fold किया जाता है (`horizontal_reduce`)। फिर:

   ```text
   acc        = BUFFER(slot, AddrSpace::Reg)                       // placeholder_like(red)
   acc_init   = STORE(AFTER(acc, input_ranges), identity)           // 0 for Add, 1 for Mul, dtype min/max for Max/Min
   acc_loop   = AFTER(acc, [acc_init, reduce_ranges..])
   body       = op(acc_loop, horizontal_inp)                        // Add/Mul/Max; float Min is -(max(-a, -b))
   store_end  = END(STORE(acc, body), reduce_ranges)   tag=TAG_MERGEABLE
   result     = AFTER(acc, [store_end])
   ```

   `input_ranges` वे ranges हैं जो input पर scope में हैं और न तो reduce हुए हैं न पहले से बंद हैं, इसलिए init घेरने वाले लूपों के अंदर आता है। कोई लूप संरचना नहीं है: `END` reduce ranges को बंद करता है और `AFTER` शृंखला डेटा निर्भरता है।
4. **`expand_horizontal_reduce`** — बिना ranges वाला `REDUCE` केवल lane fold है।
5. **END विलय** — `SINK` पर, `merge_reduce_ends` `TAG_MERGEABLE` `END`s को उनके reduce-range समूह और nesting context से समूहित करता है और हर समूह को `END(GROUP(computations), ranges)` से बदलता है; अलग nesting गहराई वाले समूहों को नए axis ids वाले cloned `RANGE`s मिलते हैं ताकि हर range ठीक एक `END` से बंद हो।
6. **`clean_up_group_sink`** — एकल-source `GROUP`s खुल जाते हैं; किसी `SINK` या `GROUP` के `NOOP`/`STACK`/`SINK`/`GROUP` sources समतल कर दिए जाते हैं।

Floats पर `Min` को `Max` के ज़रिए lower किया जाता है (`-(max(-a, -b))`) ताकि NaN वैसा ही व्यवहार करे जैसा max reduce में; integers पर यह `WHERE(a < b, a, b)` है।

## 11 — local buffers

`pm_add_local_buffers = { Stage => add_local_buffer } + movement_op_patterns` (`optimizer/mod.rs`)। इस बिंदु तक बचा हर `STAGE` वही है जिसे `pm_group_for_reduce` ने अभी बनाया है (global वाले cut पर `STORE`s बन गए, और `bufferize_to_store` ने `Local` वालों को जानबूझकर छोड़ा)। `add_local_buffer` `UOp::placeholder(max_shape, dtype, slot, opts.addrspace)` आवंटित करता है — slot group axis का `LocalBufferContext::axis_slot` है, नेस्टेड axis पथों के लिए एक नियतात्मक hash — और stage को `AFTER(buffer, [END(STORE(INDEX(buffer, ranges), compute), ranges)])` में बदलता है। फिर `movement_op_patterns` नए `INDEX` के ऊपर मौजूद किसी भी movement op को index एक्सप्रेशन में धकेल देता है।

Tinygrad भी इसी कारण local buffers जोड़ने से पहले reductions को lower करता है: grouped-reduce stage तब तक मौजूद नहीं होता जब तक `pm_reduce_local` का चरण 2 नहीं चल जाता।
