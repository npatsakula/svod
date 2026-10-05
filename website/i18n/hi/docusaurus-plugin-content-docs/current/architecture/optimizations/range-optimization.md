---
sidebar_label: Range और reduce
---

# Range और Reduce ऑप्टिमाइज़ेशन

वे नियम जो तय करते हैं कि कौन-से लूप मौजूद रहेंगे: range को विभाजित करना, मिलाना और संकरा करना, reduction को closed form में बदलना, intermediate मानों को inline या materialize करना। ये `schedule/src/rangeify/{patterns,transforms,kernel}.rs` में रहते हैं और rangeify mega-pass में, kernel cut पर तथा `apply_pre_optimization` में चलते हैं (क्रम के लिए [Rangeify](../codegen/rangeify.md) देखें)। Tinygrad: `schedule/rangeify.py`, `codegen/simplify.py`।

## Range विभाजन (`pm_split_ranges`)

`end % c == 0` वाला `RANGE % c` उस range को चिह्नित करता है; `SINK` पर हर चिह्नित range को `outer * c + inner` से बदल दिया जाता है, जहाँ `outer = RANGE(end / c)` और `inner = RANGE(c)` हैं; दोनों axis type बनाए रखते हैं और axis id `axis.child(0)` / `axis.child(1)` लेते हैं (कोई global id आवंटन नहीं)। `Warp` और `Device` range कभी विभाजित नहीं होते; image `STORE` जिस भी range से index करता है वह pinned रहता है, क्योंकि image address एक coordinate जोड़ी है, flat offset नहीं। प्रतिस्थापित graph को `symbolic + pm_fold_cast_const` से सरल किया जाता है, ताकि `inner % c → inner` और `(outer*c + inner) // c → outer` तुरंत लागू हो जाएँ।

## Range मिलाना और संकरा करना (`pm_simplify_ranges`)

`simplify_merge_adjacent` कम से कम दो range वाले हर `END` और `REDUCE` पर चलता है। `END` के लिए यह आसन्न जोड़ियाँ आज़माता है; `REDUCE` के लिए हर क्रमित जोड़ी। जोड़ी `(r0, r1)` तब मिलती है जब दोनों का axis type समान हो, end स्थिर हों, और दोनों समान `REDUCE` में दिखें (सुसंगत scoping): मिली हुई range `R(s0*s1)` `r0` को `R // s1` से और `r1` को `R % s1` से बदलती है, graph को `symbolic + pm_fold_cast_const + pm_flatten_range` से सरल किया जाता है, और merge तभी रखा जाता है जब `FloorDiv`/`FloorMod` की गिनती न बढ़ी हो (`count_divmod`, प्रति node memoized)। Symbolic end कभी merge नहीं होता: divmod गिनती नहीं बदलेगी, और symbolic गुणनफल स्थिर axis को हर बाद के const-only opt (upcast, unroll, locals, tensor cores) से छिपा देगा।

`mark_gated` हर `INDEX` से वह सीमा इकट्ठा करता है जो प्रत्येक validity clause `range < c` किसी range के लिए सिद्ध करता है; बिना guard के कहीं भी उपयोग की गई range अपने ही end पर pinned होती है, और `REDUCE` range सुरक्षित रहती हैं। `SINK` पर हर सीमित range को सबसे बड़ी सिद्ध सीमा के साथ फिर से बनाया जाता है और परिणाम सरल किया जाता है। `pm_flatten_range` के साथ (range सूचियाँ sources से पहुँचने योग्य `RANGE` से फिर से निकाली जाती हैं, `Bool`/`Void` backedge रखे जाते हैं) यही "simplify ranges" stage की पूरी सामग्री है।

## Load collapse (`pm_load_collapse`)

`reduce_load_collapse(src, ranges)`, प्रति range: range के scope के node लें (nested `REDUCE` या `STORE` मिलने पर छोड़ दें), हर बाहरी input को जो constant या `PARAM` नहीं है, एक scalar `PARAM` variable `in{n}` से बदलें जो उसका `vmin`/`vmax` रखता है (`UOp::variable`), body को उस range पर एक कृत्रिम `REDUCE(Add)` में लपेटें, और `build_reduce_load_collapse_matcher` चलाएँ। यदि कोई `RANGE` नहीं बचता, तो variables को वापस प्रतिस्थापित करें। Matcher `pm_reduce_collapse` के साथ `.or_casted()` रूप और `NE` lifting है।

सीमा नियम (`reduce_collapse_inner_patterns`, Tinygrad `simplify.py`):

| Reduce body (`r ∈ [0, N)` पर) | Closed form |
|---------------------------------|-------------|
| `WHERE(r < cut, 0, v)` | `clamp(N - cut, 0, N) * v` |
| `WHERE(r < cut, v, 0)` | `clamp(cut, 0, N) * v` |
| `WHERE(r >= lo & r < hi, v, 0)` | दो-तरफ़ा clamp गुणा `v` |
| `WHERE(idx != r, 0, e)`, `WHERE(idx == r, e, 0)` (gather) | `WHERE(0 <= idx < N, e[r := idx], 0)` |

(clamp के अंदर `min` को `-max(-a, -b)` लिखा जाता है ताकि `Max` सीमा नियम सीमांत मामलों को बंद कर सके।) इनके आसपास: `pm_reduce_unparented`; वे lifting transforms जो सीमाओं को उजागर करते हैं — `(x + y) < c → x < c - y` और `(x*y) < c → x < ceil(c/y)`, `CAST` के माध्यम से भी, `>=` और `==` इसी तरह, load-collapse संस्करण में `!=`; वितरणात्मक `sum(x + y) → sum(x) + sum(y)`; `x * bool.cast() → WHERE(bool, x, 0)`; `try_param_factor` उस condition के लिए जो range-मुक्त `PARAM` clause और range clause का AND है। बाहरी `pm_load_collapse` lift किए गए `(x + y) < c` को भी पलट देता है जब `x` में load हो, ताकि loaded indices कभी overflow न हों। संकरे matcher (बिना `!=` lifting) वाला वही engine `reduce_collapse` है, जिसे mega-pass में `pm_reduce_simplify` `num_axes == 0` वाले `REDUCE(Add)` के लिए उपयोग करता है।

```text
sum(1 for k in 0..64 if k >= length)   →   max(0, 64 - length)
```

## Reduce unparented और factor hoisting (`pm_reduce_simplify`)

`pm_reduce_unparented`: जिस reduce range को body संदर्भित नहीं करती उसे हटा दिया जाता है — `Add` परिणाम को extent से गुणा करता है, `Mul` उसे extent की घात तक ले जाता है, `Max` range को छोड़ देता है; `Min` match नहीं होता। `reduce_mul_chain`: `REDUCE(a * b * .., Add | Max)` में वे गुणनखंड जो किसी reduce range पर निर्भर नहीं हैं बाहर चले जाते हैं (`Max` के लिए केवल सिद्ध रूप से अऋणात्मक वाले), केवल integers। दोनों `POST_OPT_SYM` (stage 08) और `sym` tier में भी चलते हैं।

## Buffer हटाना (`pm_remove_bufferize`)

`INDEX(STAGE(src, ranges, opts), indices)` को stage range को consumer के indices से प्रतिस्थापित करके inline किया जाता है (`substitute_gated`; `CONST` range और `Invalid` indices छोड़े जाते हैं), सिवाय इसके कि:

1. `src` एक always-run op हो (`CONTIGUOUS`, `COPY`, `NOOP`) या stage हटाने योग्य न हो (एक `COPY` consumer, हमेशा contiguous source, multi-consumer realize सीमा, custom-kernel input);
2. compute तीन से अधिक अलग buffers पढ़ता हो (`AFTER` buffers, global `STAGE`, `MSTACK`, `PARAM`/`BUFFER`), जो kernel की argument सूची को फुला देगा;
3. compute के अंदर कोई `REDUCE` buffer पढ़ता हो (`PARAM`, `BUFFER` या `STAGE`) — inline करने से हर iteration पर read दोबारा चलेगा (`argmax(-x)` `x` को एक बार के बजाय N बार load करेगा)। ऐसे मानों पर reduce जो किसी buffer को नहीं छूते, अब भी inline योग्य है।

प्रतिस्थापन के बाद दो cleanup नियम आते हैं: `STORE(x, x)` → `NOOP`, `END(NOOP)` → `NOOP`।

`buffer_folding`: `STAGE(CONST)`, `INDEX(CONST)`, `COPY(CONST)` और `INDEX(MSTACK(CONST, ..))` constant में fold होते हैं; समान range वाला `INDEX(STAGE(compute, ranges), ranges)` stage shape तक shrink किया गया `compute` है, tags merged।

`dead_axis_removal`: एक हटाने योग्य `STAGE` (`AFTER` या always-run op पर नहीं, कोई symbolic end नहीं) उन range को छोड़ देता है जो `CONST` हैं या compute द्वारा अप्रयुक्त हैं, फिर size-1 dims को `RESHAPE` से वापस जोड़ता है और मूल shape तक `EXPAND` करता है। एक stage शून्य range के साथ समाप्त हो सकता है; उसका अस्तित्व फिर भी ज़रूरी है, वरना cut पर कोई `STORE` नहीं बनेगा।

## दो-चरणीय reductions (`split_reduceop`)

सबसे पहले rewrite में, एक tensor-form `REDUCE` जिसका input/output अनुपात `SplitReduceOpConfig::split_threshold` (32768) तक पहुँचता है, विभाजित किया जाता है: एक reduced dimension जो broadcast नहीं है (`detect_expanded_dimensions`) और `[8, 256]` में किसी divisor से विभाज्य है (सबसे बड़ा पहले) ऐसा कि intermediate output `2^22` elements से कम रहे, उसे `[.., divisor, rest, ..]` में reshape किया जाता है, मूल axes पर reduce किया जाता है, `CONTIGUOUS` से materialize किया जाता है, और divisor axis पर फिर से reduce किया जाता है। तब पहले चरण के पास parallelize करने के लिए `divisor` outputs होते हैं; दूसरा छोटा होता है।

## Grouped reductions

यह rangeify नियम नहीं है, पर उसी परिवार का है: `GROUP`/`GROUPTOP` opts `Reduce` axis के एक भाग को `GroupReduce` में बदलते हैं, और `pm_group_for_reduce` (stage 10) उसे local memory में staged एक आंशिक `REDUCE` में lower करता है, जिसे नए `Reduce` loops (`axis_id.group_reduce_loop()`) से वापस पढ़ा जाता है और फिर से reduce किया जाता है। [expander पृष्ठ](../codegen/expander.md) देखें।
