---
sidebar_label: Flash Attention
---

# Worked Example: Flash Attention

Flash Attention वह कर्नेल है जो `tk` के होने को सही ठहराता है — वही जिसे [अवलोकन](./overview) ने एक
single schedulable reduction के रूप में *not* expressible बताया था, और जिसकी वजह से एक hand-authoring
surface का होना ज़रूरी हुआ। यह chapter इसी से होकर चलता है: इसे मुश्किल क्या बनाता है, tile abstractions
इसका जवाब कैसे देते हैं, और [layout का बँटवारा](./wave-portability) असल में कहाँ सामने आता है।

हम `tk/src/kernels/fa.rs` (`build_fa_mw_rdb`) के forward कर्नेल की बात कर रहे हैं, जहाँ तक USE-चेहरे के
`flash_attention(q, k, v)` और `flash_attention_with(q, k, v, opts)` से पहुँचा जाता है। यह
[target table](./kernel-library#targets) की हर family के लिए बना है: CDNA3, RDNA3.5, RDNA4, CUDA `sm_80+` और
Apple7+।

---

## attention को autotune क्यों नहीं किया जा सकता

Plain attention `softmax(QKᵀ) · V` है। naively लिखें तो इसका मतलब है: पूरी `N×N` score matrix बनाओ,
उसे softmax करो, फिर `V` से multiply करो। पर यह score matrix बहुत बड़ी होती है और इसे कभी एक साथ पूरी
मौजूद रहने की ज़रूरत भी नहीं — इसीलिए Flash Attention keys और values के blocks पर stream करता है और
softmax को *incrementally* बनाए रखता है।

यही शब्द — incrementally — असली पेच है। softmax normalization *सभी* keys पर के maximum और sum पर निर्भर
करता है, पर हमें एक बार में सिर्फ़ एक block दिखता है। इसलिए हम running statistics रखते हैं और चलते-चलते
नतीजे को ठीक करते जाते हैं। यही **online softmax** है, और यह एक recurrence है: हर KV block वही state
पढ़ता और अपडेट करता है जो पिछले block ने बनाया था।

optimizer का action space तो बस इतना है — "इस `REDUCE` को tile और unroll करो।" पर यहाँ tile करने लायक़
कोई `REDUCE` है ही नहीं — यहाँ तो एक loop है जिसकी body अपने ही पिछले iteration पर निर्भर करती है। search
इसे ढूँढ नहीं सकती। आपको इसे ख़ुद लिखना पड़ता है।

---

## algorithm, tiles में

एक workgroup आठ waves (`NUM_WARPS`) का होता है, एक `(head, q-block, batch)` triple — launch grid
`[H, N / (q_blk · 8), B]` है। हर wave queries की एक `q_blk × D` tile की मालिक होती है, जो पूरे कर्नेल भर
registers में रहती है; आठों shared memory में एक K/V block साझा करती हैं, जिसे वे मिलकर भरती हैं। `kv_blk`
keys के हर KV block के लिए wave यह body run करती है, पूरी की पूरी tiles में:

```text
for each block of K, V:                          ┌─ everything here is a tile op
    S   = Q · Kᵀ                                 │  mma_atb into a zeroed f32 accumulator
    S   = S · log2(e)/√D                         │  the softmax scale, on the f32 scores
    S   = mask(S)                                │  causal + key-padding + segment masks
    m'  = max(m, colmax(S))                      │  update running max  (cross-lane reduce)
    P   = exp2(S - m')                           │  rescale to the new max (base-2 exp)
    l   = l · exp2(m - m') + colsum(P)           │  update running sum
    O   = O · exp2(m - m') + P · V               │  rescale accumulator, accumulate (mma_atb)
    m   = m'                                     │
O = O / l                                        └─ final normalize, transpose, store
```

हर block पर दो matrix multiplies (`Q·Kᵀ` और `P·V`), दो cross-lane reductions (max और sum), और जब भी
running max हिलता है, output accumulator का एक rescale। वह `exp2` — base-2 exponential — जान-बूझकर है, ताकि
hardware का तेज़ `exp2` unit सीधे काम आ सके। Scale `log2(e)/√D` पहले ही `Q` में fold करने के बजाय f32 score
accumulator पर apply होता है: `Q` को scale करने से वह 16-bit operand dtype में दूसरी बार round होता, और वह
error scores में उनके अपने magnitude के अनुपात में घुसता, जिसे फिर `exp2` बढ़ा देता है।

इनमें से हर line tiles पर एक `Group` operation है (`tk/src/kernels/fa.rs` में `fa_qk` और `fa_softmax_pv`)।
Score tile column-major `(KV, Q)` है, इसलिए softmax `col_reduce` से उसकी *height* पर reduce करके एक per-query
`RV` बनाता है, और rescales [Builder API](./builder-reference) वाली operator sugar हैं:

```rust
let max_vec_last = warp.copy(lp.reinit(max_vec_last), &max_vec);
max_vec = warp.col_reduce(max_vec.after(&max_vec_last), &att, |a, b| a.max(b), f64::NEG_INFINITY);
let scale_vec = (max_vec_last - &max_vec).exp2();
o_reg = o_reg * &scale_vec;
norm_vec = norm_vec * &scale_vec;
let att = (att - &max_vec).exp2();
norm_vec = warp.col_reduce(norm_vec.after(&scale_vec), &att, |a, b| a.add(b), 0.0);
```

कहीं कोई lane arithmetic नज़र नहीं आती। Recurrence एक बारीकी ज़रूरी बनाता है: running max `−∞` से नहीं,
`f32::MIN` से शुरू होता है, ताकि जो block masks किसी query row से पूरी तरह छिपा दें, वह
`−∞ − (−∞) = NaN` के बजाय `exp2(m − m') = 1` छोड़े।

---

## Streaming: double-buffered KV

यह [FLOPS कहाँ छिपते हैं](./where-flops-hide) वाला gap 2 असल काम में है। जब तक matrix core मौजूदा KV block
पर काम करता है, तब तक अगला block पहले से shared memory की ओर रास्ते में होना चाहिए। कर्नेल हर operand के लिए
**दो** LDS halves (`ker.shared_db`) रखता है और loop counter की parity से उन्हें बारी-बारी इस्तेमाल करता है:
half `kv % 2` पर compute, जबकि half `(kv + 1) % 2` load होता है।

```text
   load K/V block 0 --> LDS[A]
   ┌─────────────────────────────────────────────────┐
   │ compute on LDS[A]   ║   load block 1 --> LDS[B] │   <- overlap
   │ compute on LDS[B]   ║   load block 2 --> LDS[A] │
   │ ...                                             │
   └─────────────────────────────────────────────────┘
```

अगला block कैसे पहुँचता है, यह arch का चुनाव है, जो `Group::cp_async_fill_applies` तय करता है:

- **CUDA sm_80+**: `cp.async` global → shared बिना register staging के copy करता है। Loop का top पिछले issue को
  retire करता है (`cp_async_wait(0)` + barrier), block `kv+1` को दूसरे half में issue करता है, और मौजूदा half
  के gathers — हर 16-bit fragment के लिए एक `ldmatrix.x4` — copy के चलते-चलते run होते हैं।
- **AMD और Metal**: register-staged stream। `stage_global_to_reg` MMAs से पहले block `kv+1` के global loads
  per-lane registers में issue करता है, `commit_regs_to_local` उनके बाद उन्हें दूसरे half में लिखता है, और हर
  trip पर एक `war_fence2` barrier — जिसे gathers consume करते हैं और जो commit को dependency के रूप में ढोता
  है — अभी लिखे गए half पर read-after-write और उस half पर write-after-read, दोनों को cover करता है जिसे हर
  wave ने अभी-अभी gather करना ख़त्म किया।

Prefetch index block count के modulo wrap होता है, इसलिए आख़िरी trip operand से बाहर भागने के बजाय block 0
(जो कभी gather नहीं होता) दोबारा पढ़ता है। KV loop ख़ुद एक *dynamic* bound वाला `Loop` है: `causal` में हर
q-block सिर्फ़ `(block_q_base + 1) · 8 · q_blk / kv_blk` super-blocks पर जाता है — causal block-skip।

---

## layout की बारीकी: दो matmuls के बीच relayout

यहीं [Layouts और wave size](./wave-portability) theory से निकलकर असल में सामने आता है। कर्नेल दो
matrix multiplies करता है, और पहले का output (`S = Q·Kᵀ`, जो softmax के बाद `P` बन जाता है) दूसरे का
*input* है (`P·V`)। तो क्या score accumulator को सीधे एक operand की तरह वापस feed किया जा सकता है?

- **CDNA, RDNA4, CUDA और Metal पर** (`acc_reusable_as_input() == true`): हाँ। Accumulator और operand एक ही
  lane map साझा करते हैं — CDNA पर MFMA fragment, gfx12 का एक fragment, CUDA पर A-operand register order में
  `m16n8` C fragments, Apple पर एक `simdgroup_matrix` map — इसलिए `att_mma` f32 → 16-bit cast के साथ एक
  register `copy` है।
- **RDNA3 पर** (`acc_reusable_as_input() == false`): नहीं। even/odd accumulator और replicated operand अलग
  होते हैं, इसलिए `P` **LDS से होकर एक round-trip** करता है: accumulator का एक `store_local_fenced` एक
  per-workgroup `[8 · kv_blk, q_blk]` scratch tile (`att_smem`) में इस wave के band में, फिर operand map के
  तहत एक `load` वापस। `FaPolicy::att_band` इसे report करता है, ताकि shared-memory budget band को गिने।

कर्नेल `att_smem` allocate करते समय एक बार `ker.caps.acc_reusable_as_input()` पर branch करता है; hot loop
`Option<ST>` पढ़ता है। algorithm वही, पर दो physical realizations — ठीक वही portability tax जिसका ज़िक्र पिछले
chapter में था, और वह भी सबसे अहम कर्नेल के सबसे hot loop में।

---

## Masking

Causal masking (एक query किसी future key पर attend नहीं कर सकता), key-padding (`key_lens[b]`, valid count पर
या उसके आगे की keys को नज़रअंदाज़ करना) और packed-sequence segments (`seg_start[b, q]`, query के अपने segment
से पहले की keys को नज़रअंदाज़ करना) — तीनों softmax से पहले score tile `S` पर apply होते हैं। तीनों
`mask_where` calls हैं: हर score element की position इसी से तय हो जाती है कि उसे कौन-सा fragment और lane रखता
है, इसलिए mask tile के अपने `LaneMap` से compute होता है, fetch नहीं। Optional mask tensors trailing globals
हैं जो `o, q, k, v` के बाद सादे `ker.gl` calls से bind होते हैं, इसलिए unmasked ABI नहीं बदलता।

---

## Per-warp tile

`FaConfig { q_blk, kv_blk, unroll, causal }` body का tuning knob है। Baseline `{16, 32}` है — एक 16-row Q tile
और एक 32-key KV super-block, जो `{16, 16}` की तुलना में per-wave MMA ILP बढ़ाता है और softmax की bookkeeping
आधी कर देता है। `FaPolicy` (`tk/src/kernels/fa.rs`) हर family के लिए ऊँची tile तब चुनता है जब launch grid
device के compute units को cover कर ले और head dim family की सीमा के नीचे रहे, और ऐसे head dim को decline
करता है जिसके buffers shared memory से ज़्यादा हों (AMD पर 64 KiB, CUDA पर 48 KiB static, Apple पर 32 KiB)।
पहले launch पर चारों `FA_TILES` candidates time होते हैं और विजेता cache होता है — [Autotuning](./tuning)।

`unroll` सिर्फ़ CUDA के लिए on है: NVPTX backend accumulators को registers में तभी रखता है जब हर register
index constant हो, इसलिए body flat emit होती है (`Kernel::set_unroll`); AMD rolled रूप रखता है।

:::tip[GPU विशेषज्ञों के लिए]
compute/memory overlap को `tk` में raw scheduling intrinsics के रूप में hand-emit नहीं किया जाता, जैसा
HipKittens के कर्नेल में होता है। इसके बजाय KV loop पर `sched::pipeline(SchedKind::Attention, kv_idx)`
(`tk/src/kernels/fa.rs`) का annotation लगा होता है — in-loop K/V buffers से होकर गुज़रने वाला एक marker, जिसे
`codegen/src/llvm/sched.rs` का post-linearization pass consume करता है। Body बताती है कि *क्या* overlap करना है;
instruction ordering का फ़ैसला pass करता है। आज यह CDNA पर marker को `@llvm.amdgcn.iglp.opt(0)` में lower करता
है — backend का तैयार MFMA/memory interleave — और बाक़ी जगह उसे एक निष्क्रिय comment छोड़ देता है; attention
kind के लिए योजना एक softmax-aware comb की है जो exponential काम को matrix ops के नीचे बुन दे।
:::

---

## यह क्यों ज़रूरी है

Flash Attention पूरे section को एक ही file में समेट देता है:

- यह इसलिए है क्योंकि **online softmax एक recurrence है**, कोई tileable reduction नहीं
  ([अवलोकन](./overview));
- यह **streaming और overlap** के दम पर जीता या मरता है ([FLOPS कहाँ छिपते हैं](./where-flops-hide));
- यह पूरी तरह **tiles और roles** में व्यक्त होता है, कभी lane indices में नहीं ([Tiling क्या है](./tiling));
- यह बाक़ी सब चीज़ों जैसा **वही UOp IR** बनकर compile होता है और lazy graph में एक
  `Op::Call` के रूप में शामिल होता है ([IR में authoring](./lowering));
- और अपने hot loop में यह एक explicit **accumulator-reuse branch** साथ रखता है, हर fragment layout के लिए एक
  ([Layouts और wave size](./wave-portability))।

इसीलिए यह हाथ से लिखा गया है, और इसीलिए इसे लिखने के लिए `tk` मौजूद है। इसे isolation में run करके इसके
numbers जाँचने के लिए, देखें [डीबगिंग](./debugging)।
