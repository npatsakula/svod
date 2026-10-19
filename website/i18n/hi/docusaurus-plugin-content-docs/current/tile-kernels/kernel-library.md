---
sidebar_label: कर्नेल लाइब्रेरी
---

# कर्नेल लाइब्रेरी

`svod_tk3::kernels` का हर कर्नेल एक function `fn k<T: Elem>(spec: &Spec) -> Program` है, जो 16-bit
element type (`BF16` या `F16`) पर generic है। हर spec में एक `batch: Batch` होता है:
`Batch::Static(n)` `n` batches launch करता है, और `Batch::Var { name, min, max }` नाम से bind हुए
runtime variable की live गिनती launch करता है, buffers `max` के आकार के होते हैं। हर config type में
एक `lowering(target)` है। मॉडल इन कर्नेलों तक [op layer](./op-layer) के ज़रिए पहुँचते हैं, जो config
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

`GemmCfg { tile: [bm, bn, bk], stages, warps: [rows, cols], group_m, unroll }`। `m` से आगे की rows और
`n` से आगे के columns bounded views हैं, इसलिए `m` और `n` मुक्त हैं। `k` का `bk` का गुणज होना ज़रूरी है।

Op layer candidates चार tile families से लेती है, हर एक अपनी sm_86 पर मापी गई सबसे अच्छी pipeline
के साथ:

| Family | Stages | Warp grid | Unroll | मापी गई टिप्पणी |
|---|---|---|---|---|
| 128×128×32 | 3 | 2×4 | नहीं | 4096³ का शिखर यहीं: 25.4 TFLOP/s |
| 128×64×32 | 2 | 2×2 | हाँ | |
| 64×128×32 | 2 | 2×2 | नहीं | |
| 64×64×32 | 3 | 2×2 | हाँ | M = 704 (Nemotron): 128-row tiles 64×64 से हारते हैं |

`config::gemm_candidates` सबसे पहले वह सबसे बड़ी family रखता है जिसका grid हर SM को 8 blocks देता है
और output area का अधिकतम 1/16 pad करता है (वरना 64×64)। फिर वह उस family के variants जोड़ता है
(दूसरी stage count, `bk = 64`, उल्टा `unroll`, transposed warp grid), फिर बाक़ी families। जब `k`
`bk` का गुणज नहीं होता तो `bk` आधा होते-होते 16 तक जाता है, और जिन configs की ring target की shared
memory से बड़ी है वे हटा दिए जाते हैं। अधिकतम आठ candidates बचते हैं, और [tune store](./tuning)
उनमें से चुनता है।

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
| `AttnMask::bias` | stream type में `[batch या 1, heads, t, tk]` additive bias (row stride 8 तक rounded), हर block में पढ़ा जाता है और masks से पहले scaled scores में जोड़ा जाता है |

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

`FaCfg { bq, bkv, stages, splits }` हर 16 query rows पर एक warp इस्तेमाल करता है। Head dim के हिसाब से
candidates, पहला untuned चुनाव है:

| `d` | Candidates `(bq, bkv, stages)` | मापी गई टिप्पणी (sm_86) |
|---|---|---|
| 48, 64, 128 with `t ≤ 16` | (16, 64, 2), (16, 64, 3), (16, 32, 2) | Decoder step bandwidth-bound है: one-warp blocks हर SM पर कई blocks की जगह छोड़ते हैं |
| 48 | (64, 64, 2), (64, 64, 3) | ऐसे shapes जिनके K/V fills block के threads में बँट जाते हैं (96-byte rows) |
| 64 | (64, 64, 2), (64, 64, 3), (128, 64, 2), (64, 32, 2), (128, 32, 2), (64, 32, 3) | |
| 128 | (64, 32, 2), (64, 32, 3), (128, 32, 2), (64, 64, 2), (128, 64, 2), (128, 32, 3) | `bkv = 64` ने 18.2 TFLOP/s मापा, `bkv = 32` के 22.2 के मुक़ाबले: प्रति block 64 KB से हर SM पर एक ही block बचता है |

हर tile config को split counts से गुणा किया जाता है। `Attn::splits = Some(n)` `n` तय करता है, key
block count पर सीमित। `None` के साथ, `config::split_candidates` 1, 2, हर SM पर दो blocks के आसपास की
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
