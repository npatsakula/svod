---
sidebar_label: कर्नेल लाइब्रेरी
---

# कर्नेल लाइब्रेरी

`svod_tk3::kernels` का हर कर्नेल एक function `fn k<T: Elem>(spec: &Spec) -> Program` है, जो 16-bit
element type (`BF16` या `F16`) पर generic है। हर spec में एक `batch: Batch` होता है:
`Batch::Static(n)` `n` batches launch करता है, और `Batch::Var { name, min, max }` नाम से bind हुए
runtime variable की live गिनती launch करता है, buffers `max` के आकार के होते हैं। हर config type में
एक `lowering(&device)` है (एक `atoms::Device`: target और उसकी costs)। मॉडल इन कर्नेलों तक [op layer](./op-layer) के ज़रिए पहुँचते हैं, जो config
चुनती है।

| कर्नेल | Spec → program | कर्नेल का नाम | Op |
|---|---|---|---|
| GEMM + epilogue | `GemmSpec` → `gemm::gemm` | `gemm` | `ops::linear` |
| Implicit-GEMM convolution | `ConvSpec` → `conv::conv` (+ split होने पर `gemm::split_merge`) | `conv`, `split_merge` | `ops::conv2d` |
| Flash attention forward | `AttnSpec` → `attention::attention` | `flash_attention` | `ops::attention` |
| Split merge | `CombineSpec` → `attention::combine` | `combine_splits` | `splits` के साथ `ops::attention` |
| Attention prologue | `HeadsSpec` → `heads::heads` | `heads` | `ops::heads` |
| LayerNorm / RMSNorm | `NormSpec` → `rows::norm` | `layer_norm` / `rms_norm` | `ops::{layer_norm, rms_norm, add_*}` |

## GEMM {#gemm}

`c = act(a·bᵀ + bias) + residual`, जहाँ `a [batch·m, k]` और `b [n, k]`। Matrix product के बाद का
सब कुछ f32 accumulator पर चलता है, और नतीजा store पर एक बार round होता है।

| `Epilogue` field | असर |
|---|---|
| `bias` | accumulator में जोड़ी गई `[n]` row (gated होने पर `[2n]`) |
| `act` | `Act::None`, `Act::Gelu` (erf approximation, error अधिकतम 1.5e-7), `Act::Silu` |
| `gated` | `b` `[2n, k]` है, gate rows ऊपर, up rows नीचे; output `act(gate)·up` (SwiGLU, GeGLU) |
| `residual` | `[batch·m, n]`, सबसे आख़िर में जोड़ा जाता है |

`GemmCfg { tile: [bm, bn, bk], stages, warps: [rows, cols], group_m, unroll, core, crews }`। `m` से आगे की rows और
`n` से आगे के columns bounded views हैं, इसलिए `m` और `n` मुक्त हैं। `k` का `bk` का गुणज होना ज़रूरी है।

Op layer कोई tile table नहीं रखती। `config::Planner::gemm_ranked` construct के knobs गिनता है (32–256
rows और 48–256 columns के output tiles, `bk` 16–64, नौ warp grids, वे ring depths जो target का prefetch
चलाता है, rolled और unrolled), वही रखता है जिसे lowering चला सके (atoms warp grid को tile करते हैं, fills
हर thread पर पूरे 16-byte chunks में बँटते हैं, ring shared memory में समाती है) और बाक़ी को device की मापी
गई `Costs` से SM time के nanoseconds में rank करता है। एक block का trip तीन pipes से issue होता है: हर wave के
operand fragments जो shared memory से पढ़े जाते हैं, matrix-core steps और stream होकर आती fill, साथ में trip
की fixed cost। ये pipes उन waves के बीच overlap होती हैं जो एक SIMD रखता है: सबसे लंबी पूरी चुकाई जाती है,
बाक़ी हर SIMD की waves पर एक का हिस्सा, ज़्यादा से ज़्यादा चार तक (`OVERLAP_WAVES`: gfx1201 और sm_86 पर हर
SIMD की सोलह waves एक tile को चार से बेहतर नहीं चलातीं)। जो SIMD `step_lone / step` से कम waves रखता है
(उतनी waves matrix-core chain की latency छिपाती हैं), वह steps उतने गुना धीमा चलाता है, और एक wave के
independent accumulators waves की तरह गिने जाते हैं। सबसे व्यस्त SM `per_sm = ⌈blocks / sms⌉` blocks का
काम issue करता है, एक के ख़त्म होते ही अगला उठाते हुए, और trip की fixed cost उन blocks में बँटती है जो वह एक
साथ रखता है (`resident`, shared memory और register estimate से); उन blocks का एक round कम से कम एक wave का
अपना issue है (जिस block में SM के SIMDs से कम waves हैं वह बाक़ी को idle छोड़ता है), और हर round अपने issue के
ऊपर वह memory latency चुकाता है जिसे एक pipeline stage आगे छिपा नहीं सकता:
`cost = trips × (max(per_sm × (body + trip / resident), rounds × simds × (body + trip) / waves) + rounds × latency / (stages − 1))`
जहाँ `rounds = ⌈per_sm / resident⌉` और `body` overlap की हुई pipes हैं (`Planner::pipelined`)। Fill का byte
इस हिसाब से priced है कि वह कहाँ से आता है: distinct bytes cache के तीन-चौथाई में समाएँ तो L2 दर, उससे डेढ़
गुना से ज़्यादा हों तो DRAM दर, बीच में दोनों का मिश्रण, और कभी उस हिस्से से महँगा नहीं जिसे एक साथ चल रहे
blocks एक-दूसरे की lines से दोबारा पढ़ते हैं — अपने group की A rows और अपनी B columns (`Planner::fill_byte`)। Wave वाले bound के बिना gfx1151 की ranking one-warp 256×256 tiles से
शुरू होती थी जो
1–3 TFLOP/s पर चलते थे। Layout की register count सिर्फ़ अनुमान लगाती है कि कितने blocks resident हैं; यह
कुछ भी admit नहीं करती, क्योंकि जहाँ fragments replicate होते हैं (gfx11 WMMA) वहाँ यह ज़्यादा गिनती है।
Ranking किसी tile की हर pipeline रखती है; [tune store](./tuning) उन पहले 64 output tiles और warp grids में
से चुनता है जिनके compiled kernels spill नहीं करते, हर एक अपनी पहली clean pipeline पर (सब spill करें तो:
पहले 64 जो compile हों), फिर विजेता tile की बाक़ी pipelines में से। Admit compiler करता है और चुनाव घड़ी,
क्योंकि model block tiles को अच्छा और warp grids को ख़राब क्रम देता है (sm_86 पर मापा गया सबसे तेज़ config
109 tiles में 92वें स्थान पर था); pipelines को विजेता मिलने तक वह अकेला rank करता है, इसलिए spill करने वाली
unrolled ring अपने
rolled जुड़वाँ के लिए जगह छोड़ती है: gfx1151 पर d = 128 के unrolled 64×16 और 128×16 attention tiles spill
करते हैं और rolled वाले clean compile होते हैं। Model के लिए gated weight एक `2·bn`-चौड़ा B tile और दो
accumulators है, इसलिए उसकी बढ़त आधी चौड़ाई पर आती है। किसी device का `Planner` उसका target, उसकी मापी गई
costs और वे problems रखता है जिन्हें उसने rank किया है (op हर call पर अपना shape plan करता है; lattice को
rank करने में एक millisecond लगता है), process में हर device के लिए एक planner।

Model में कोई fitted constants नहीं हैं: हर cost device की अपनी है, एक probe से मापी गई। जहाँ ranking और
कोई माप असहमत हों, माप जीतता है: shape को `qwen3_gemm_candidates_probe` में जोड़ें और देखें कि ranking
कौन सी cost चूकती है।

## Flash attention {#flash-attention}

Sequence-major `q [batch, t, heads, d]` और `k`, `v [batch, tk, kv_heads, d]` पर
`o = softmax(q·kᵀ·scale)·v`। Q registers में रहता है और K/V एक shared ring से stream होते हैं।
Online-softmax state `(m, l, o)` f32 में `exp2` के साथ carry होता है।

| Feature | कैसे |
|---|---|
| GQA | Query head `h` KV head `h / (heads / kv_heads)` पढ़ता है |
| Cross attention | `tk ≠ t` views का गुण है |
| कोई भी `t`, `tk` | Padding की जगह bounds: query tile और आख़िरी key block bounded views हैं, और `tk` से आगे की keys mask होती हैं |
| Head dims | 48, 64, 128 |
| `AttnMask::causal` | Diagonal से आगे के key blocks trip count के ज़रिए छोड़ दिए जाते हैं |
| `AttnMask::window` | हर query के आसपास `(left, right)`; बाहर के blocks छोड़े जाते हैं |
| `AttnMask::key_lens` | `[batch]` i32 वैध key गिनतियाँ; लंबाई से आगे के blocks छोड़े जाते हैं |
| `AttnMask::key_mask` | `[batch, tk]` i32 दिखने वाली keys (row stride 8 तक ऊपर round), हर block में पढ़ी जाती है |
| `AttnMask::seg_start` | packed rows के `[batch, t]` i32 segment starts, `t` के साथ non-decreasing |
| `AttnMask::bias` | stream type में `[batch या 1, heads, t, tk]` additive bias (row stride 8 तक rounded), K और V के बगल में अपने ring slot से stream होता है (admission में इसके bytes उन्हीं की तरह गिने जाते हैं) और masks से पहले scaled scores में जोड़ा जाता है; appended rows का block अपना bias tile सीधे पढ़ता है |

Masks tile-class का काम हैं। सिर्फ़ वे key blocks predicate गिनते हैं जिन्हें causal, window, length
या segment का कोई किनारा काटता है (block start पर `select_if`); पूरी तरह अंदर वाले blocks बिना mask
चलते हैं। Bool key mask हर block में लगता है। जिस query को कोई key नहीं दिखती वह NaN देती है, जैसे
ख़ाली row पर softmax देता है।

**Cache mode** (`AttnSpec::cache = Some(Cache { .. })`) K और V को एक cache
`[rows, tk, heads_total, d]` से पढ़ता है जिसमें कई layers के heads होते हैं:

| `Cache` field | असर |
|---|---|
| `head_start` | Cache row में इस attention के `kv_heads` heads में से पहला |
| `row_map` | एक `[batch]` i32 parameter: हर batch lane कौन सी cache row पढ़ती है |
| `appended` | Cached prefix के बाद score होने वाली `[batch, kv_heads, d]` key और value (वह token जिसे decoder step ने अभी project किया) |

**Key splits** (`FaCfg::splits > 1`) कुछ लंबी rows को, जैसे decoder step को, ज़्यादा blocks देते हैं।
हर split f32 partials `o_part`, `m_part` और `l_part` लिखता है, और `combine` उन्हें मिलाता है।

`FaCfg { bq, bkv, stages, splits, unroll, warp_rows, .. }` हर `warp_rows` query rows पर एक warp इस्तेमाल करता है: 16 (एक atom row) या 32 (warp का पढ़ा हर key और value fragment दो atom rows के products को खिलाता है, प्रति lane दुगने accumulators की क़ीमत पर)। 32-row रूप अपने 16-row tile का एक variant है: यह 64 या ज़्यादा queries के blocks के लिए दिया जाता है जहाँ यह उतने ही blocks resident रखता है, shortlist में कभी tile का प्रतिनिधित्व नहीं करता, और tuner इसे दावेदार tiles के pipelines के साथ मापता है। Candidates
GEMM वाले ही lattice और cost model से आते हैं (`Planner::attention_ranked`): 16–128 rows के query blocks,
16–128 keys के key blocks, वे ring depths जो target चलाता है, rolled और unrolled, उतने तक सीमित जो lower हो
(fills हर thread पर chunks में बँटते हैं, ring shared memory में समाती है) और इस हिसाब से rank होते हैं कि
एक key block की cost क्या है (हर warp के K और V fragments पढ़े जाते हैं, दोनों tiles stream होकर आते हैं,
matrix-core steps, हर score पर softmax का vector काम और trip की fixed cost), query blocks के हर round के
key blocks पर, हर round GEMM की तरह bound। Register अनुमान query, key, score, probability और value
fragments हैं, जो compiled loop में साथ-साथ live रहते हैं; कोई config spill करता है या नहीं, यह compiled
kernel बताता है। Problem की queries से आगे का warp अपनी padded rows का काम जोड़ता है, इसलिए जहाँ tiles device
को भर देते हैं वहाँ decoder step one-warp blocks से शुरू होता है; जहाँ वे SIMDs को idle छोड़ते हैं वहाँ padded
warp उनमें से एक पर चलता है। Causal mask हर query block से उसका diagonal वसूलता है। जहाँ value tile का fill
registers से होकर जाता है वहाँ वह column-major रखा जाता है (देखें
[Layouts and Lowering](./layouts-and-lowering#layouts-and-lowering)), इसलिए उसके fragments keys की तरह
16-byte loads के रूप में इकट्ठा होते हैं; gfx1201 पर मापा गया तो यह row-major tile के हर key पर एक load के
मुक़ाबले प्रति shape 5–28% है। RDNA4 पर fill ख़ुद एक transposing load (`global_load_tr_b128`) है, इसलिए
उसके registers column runs रखते हैं और वे भी 16-byte vectors के रूप में store होते हैं।

Softmax step जितना कम vector काम कर सके करता है, जो वहाँ सबसे ज़्यादा मायने रखता है जहाँ matrix cores
SIMD के ALUs साझा करते हैं (RDNA)। कोई row नया running maximum तभी लेती है जब वह log2 इकाइयों में 8 से
ज़्यादा बढ़े (FlashAttention-4 का lazy rescale): वरना correction ठीक 1 रहता है और probabilities 2⁸ से नीचे
रहती हैं। Planner हर config सादे forms में देता है; बाक़ी पाँच (`FaCfg::forms`: `truncate_p`, `skip_rescale`, `raw_max`,
`sum_after`, `bias_direct`) tuner knobs हैं, जो tiles के साथ बारी-बारी tune होते हैं (`ops::attention::tune`, `descend` के ज़रिए), क्योंकि
कोई tile केवल कुछ forms के साथ ही clean compile हो सकता है: ranking के पहले clean compile होने वाले config पर
एक forms pass, फिर उसके रखे forms के साथ ranking पर tile और split का चुनाव, फिर उस winner पर दोबारा एक
forms pass (store-backed, इसलिए कुछ न बदले तो सस्ता)। Forms pass एक coordinate descent है: हर form एक बार,
इसी क्रम में, एक two-candidate tune जिसमें अब तक के forms का मुक़ाबला उसी से होता है
जिसमें वह form पलटा गया हो, और पलटाव तभी रखा जाता है जब वह तेज़ मापे (spill करने वाला पलटाव admit नहीं
होता)। कौन सा form registers बचाता है, यह कहीं नियम नहीं है: माप तय करता है। Tuning बंद होने पर हर attention
सादे forms चलाता है। `truncate_p` के साथ bf16 में रखी probabilities round नहीं, truncate होती
हैं — जहाँ rounding कई instructions लेती है वहाँ एक instruction में एक जोड़ी — और row sum उन्हें वैसे ही जोड़ता है
जैसे वे रखी हैं, ताकि normalization unbiased रहे; जो targets एक जोड़ी को एक instruction में round करते हैं
(sm_80+, gfx950) वे f32 मान round और sum करते हैं, क्योंकि रखे मानों का अतिरिक्त conversion sm_86 पर
3–5% महँगा था। `skip_rescale` के साथ output accumulator का rescale एक `Scale` है, जिसे हर वह wave छोड़ देती है
जिसकी सभी rows ने अपना maximum रखा; इसका फ़ायदा वहाँ हर जगह मिला जहाँ matrix products vector ALUs पर issue
होते हैं, और अलग matrix cores के बगल में केवल वहाँ जहाँ भरे tile की lane 64 output registers रखती है (sm_86:
d = 128 पर +4%, d = 64 पर neutral और decoder step पर 5–7% धीमा, जिसकी एक row सादा multiply रखती है)।
`raw_max` के साथ (queries से भरे RDNA tiles पर फ़ायदेमंद), running maximum raw scores की इकाइयों में रहता है
(positive scale maximum के साथ commute करता है; bias scaled form रखता है), तो केवल वे दो row मान scale होते
हैं जो exponent और correction लेते हैं — हर score की जगह हर row पर एक multiply — step की dependent chain पर
एक multiply की कीमत पर, जिसे decoder step की अकेली wave महसूस करती है (3%) और भरा tile नहीं। `sum_after` के
साथ row sum rescale से पहले आने की जगह product के बाद आता है; truncated probabilities दोनों तरह वैसे ही जोड़ी
जाती हैं जैसे रखी हैं (bf16 pair dot product, जहाँ target के पास वह है), rounded वाली f32 में। Product के बाद
दोनों consumers truncated मान एक बार packed पढ़ते हैं (gfx1201); उससे पहले, sm_86 पर rounding के साथ, यह क्रम
d = 128 पर 3% तेज़ था (24.7 बनाम 24.0 TFLOP/s, इस काम से पहले 22.4) और d = 64 पर तथा decoder step पर
neutral। `bias_direct` के साथ (जो सिर्फ़ bias के तहत flip होता है) हर key block का bias tile K और V के बगल
वाले अपने ring slot से stream होने की जगह, जो planned form है, scores के layout में सीधे global memory से पढ़ा
जाता है: gfx1151 पर, जिसका accumulator हर lane में एक column रखता है और ring के tile को दो-दो byte में gather
करता है, causal mask के बिना direct reads तेज़ निकले (d = 64, B 4, H 8, T 2048 पर 3.0 बनाम 3.6 ms) और उसके
साथ ring तेज़ (1.88 बनाम 2.12 ms)। Causal mask (या हर पहले वाले को देखने वाले chunks) के नीचे
आख़िरी query blocks, जो सबसे ज़्यादा keys score करते हैं, पहले चलते हैं, ताकि round की पूँछ उसके सबसे छोटे
blocks हों। gfx1201 पर lazy rescale, truncation, skip और नीचे के raw `exp2` ने d = 128 kernel को random data
पर 90 से 116 TFLOP/s और Qwen3 के causal attention को 63 से 77 तक ले गए; raw maximum, pair dot product और
block order उन्हें 120 और 79 तक, d = 128 causal 96 → 100 और d = 64 105 → 108 (causal 87 → 92) तक ले गए,
decoder step अपरिवर्तित। Loop के अकेले WMMAs जिस 162 TFLOP/s पर चलते, उसके मुक़ाबले जो बचा है वह पतला फैला है:
vector काम 15%, बाक़ी loads, stores, barriers और addressing।

हर tile config को split counts से गुणा किया जाता है। `Attn::splits = Some(n)` `n` तय करता है: सिर्फ़
वे configs rank होते हैं जिनमें कम से कम `n` key blocks हों, `n` पर (उस सबसे बड़ी key block count पर
सीमित जो किसी config में हो), इसलिए बताई गई गिनती किसी बड़े key block की वजह से कभी नहीं छूटती। `None` के
साथ, `Planner::splits` 1, 2, हर SM पर दो blocks के आसपास की
गिनतियाँ, और हर key block के लिए एक split सुझाता है। यह सिर्फ़ उन्हीं गिनतियों को रखता है जो key
block count और उस SM लक्ष्य के दोगुने में से छोटे तक हों। Unsplit
config पहले आता है, और split candidate अपने merge के साथ मिलाकर मापा जाता है।

Attention throughput probe (B 4, H 8, T 2048, bf16) ने tk1 के मुक़ाबले TFLOP/s में मापा:

| Case | tk3 | tk1 |
|---|---|---|
| d 64 | 24.1 | 23.0 |
| d 64 causal | 22.6 | 20.7 |
| d 128 | 22.3 | 22.3 |
| d 128 causal | 20.9 | 16.5 |

Decode probe एक Whisper large-v3 decoder step के attention का समय µs में मापता है। Whisper का
decoder step इन्हीं कर्नेलों को बुलाता है:

| Case | tk3 | tk1 |
|---|---|---|
| Self attention, 200 cached keys + appended एक | 21.5 | 89.1 |
| Cross attention, 1500 shared keys | 66.3 | 78.8 (636 unsplit) |

खुला मुद्दा: decode `row_map` अब भी tuned shape से बाहर है (tuner हर lane पर एक row वाले scratch caches पर
मापता है, जहाँ map कचरा rows पढ़ता), इसलिए tuned cross attention अपने map के साथ सबसे तेज़ न भी हो: gfx1151
पर (2026-10-14) वह 60 µs unsplit के मुक़ाबले 90 µs पर tune हुआ।

## Attention prologue {#attention-prologue}

`heads` एक fused projection `qkv [batch, t, (heads + 2·kv_heads)·d]` को sequence-major
`q [batch, t, heads, d]` और `k`, `v [batch, t, kv_heads, d]` में बाँटता है। वैकल्पिक रूप से यह `q`
और `k` को head पर `[d]` weights के साथ RMS-normalize करता है, फिर उन पर rotary embedding लगाता है।
Rotation head के दोनों हिस्सों को जोड़ता है, `[t, d/2]` की `(cos, sin)` tables से, जो साझा हों या हर
batch की अलग (`Rope { per_batch }`)। एक block एक head slot की `br` rows संभालता है। `d` 16..=256 में
दो की घात होना चाहिए, और candidates `br ∈ {4, 8, 16}` हैं।

## Norms {#norms}

`rows::norm` हर row पर एक warp और f32 में एक fused reduce-then-map इस्तेमाल करता है:

| `Norm` | सूत्र |
|---|---|
| `Layer` | `(x − mean)·rsqrt(var + eps)·w + b` |
| `Rms` | `x·rsqrt(mean(x²) + eps)·w` |

`residual: true` के साथ यह element type पर round किए गए `x + residual` को normalize करता है और वह sum
भी लिखता है, जो transformer layer की pre-norm residual stream है। `d` 256..=2048 में दो की घात होना
चाहिए, और candidates प्रति block `br ∈ {4, 8, 16}` rows हैं। sm_86 पर memory bandwidth का 92% मापा गया।
