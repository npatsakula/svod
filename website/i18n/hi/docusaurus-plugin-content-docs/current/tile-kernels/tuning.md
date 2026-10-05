---
sidebar_label: Autotuning
---

# पहले इस्तेमाल पर Autotuning

किसी कर्नेल की tile table एक search space है, जवाब नहीं। GEMM हर family के लिए तीन से पाँच tiles रखता है,
flash attention चार per-warp tiles, single-query attention कुछ K/V splits — और कौन-सा जीतता है, यह shape,
device के compute units की गिनती और उसकी clock पर निर्भर करता है। इसलिए जब कोई device पहली बार किसी shape से
मिलता है, `svod-tk` हर फ़िट होने वाले candidate को compile और time करता है, सबसे तेज़ को रखता है, और उसे disk
पर याद रखता है। `tk/src/tune.rs` ही पूरा mechanism है।

---

## पहले launch पर क्या होता है

`GemmPolicy::tuned`, `FaPolicy::tuned` और `SqPolicy::tuned` (`tk/src/kernels/` में) `TuneStore::select` के
ज़रिए एक ही क्रम चलाते हैं:

1. **Filter:** table को उन candidates तक छाँटें जो shape को tile करते हैं (और, GEMM के लिए, माँगा गया
   `Epilogue` रखते हैं)। एक candidate या कोई नहीं: static चुनाव लौटाएँ, कुछ measure न करें।
2. **Memo:** एक process-wide `HashMap<TuneKey, usize>` दोहराए गए shape का जवाब बिना कर्नेल build किए देता
   है — एक plan हर node पर एक ही shape एक बार पूछता है।
3. **Store:** memo miss पर, हर candidate का `SINK` placeholder buffers के ख़िलाफ़ build करें और उसका
   fingerprint लें (`kernel_fingerprint`)। Digests store line में जुड़ते हैं, इसलिए कर्नेल body में बदलाव
   दोबारा measure करवाता है। Device की file में line खोजें।
4. **Measure:** store miss पर, हर candidate को shape के synthetic operands (`Tensor::randn`, dtype में cast,
   device पर moved) पर `compile_kernel` करें। जो पहला build हो गया, वह clock उठाता है — `warm_clock` उसे तब
   तक dispatch करता है जब तक उसका समय गिरना बंद न हो जाए या 1.5 s न बीत जाएँ; पहले से load में चल रहा device
   कुछ ही runs में plateau पर पहुँच जाता है। फिर `round_robin_min` तीन rounds तक हर candidate को बारी-बारी
   time करता है और हर एक का minimum रखता है, ताकि किसी को ऐसी clock पर न आँका जाए जिस पर बाक़ी नहीं थे।
5. **Keep:** सबसे तेज़ को memo में और store file में रखें। जो candidate build या dispatch नहीं हो पाता, उसे
   छोड़ दिया जाता है; अगर कोई नहीं हो पाता, तो कुछ cache नहीं होता और static चुनाव इस्तेमाल होता है।

File तक सिर्फ़ measure किया गया विजेता पहुँचता है। `select_with` वही policy है जिसमें measurement caller
देता है, और `tk/src/test/unit/tune.rs` के unit tests इसी तरह उसे बिना GPU के चलाते हैं।

---

## Key और store

```rust
// tk/src/tune.rs
pub struct TuneKey {
    pub kernel: &'static str,   // "gemm_nt", "flash_attention", "sq_attention"
    pub device: String,         // "<arch target name>-<compute units>cu"
    pub shape: Vec<usize>,      // the kernel's own shape tuple, dtype width and flags included
    pub config: u64,            // a digest of the candidate set (and anything else the graphs vary with)
}
```

GEMM की key `[m, k, n, dtype.bytes(), epilogue.code()]` है; flash attention की
`[b, n, h, h_kv, d, causal, mask.code(), dtype.bytes()]`। Table बदलने से `config` बदलता है, इसलिए नया
candidate दोबारा measure होता है।

Store हर device और crate version के लिए एक file है, हर entry के लिए एक line:

```text
<kernel>|<device>|<shape>|<builds digest> <winning index> <ns>
```

इनमें से पहली जगह पर:

| जगह | कब |
|---|---|
| `$SVOD_TK_TUNE_DIR/` | variable set हो |
| `$XDG_CACHE_HOME/svod/tk_tune/` | वरना, जब `XDG_CACHE_HOME` set हो |
| `$HOME/.cache/svod/tk_tune/` | अन्यथा |

File का नाम device string है जिसमें non-alphanumerics बदल दिए जाते हैं
(`gfx1201_64cu-v0.1.0.txt`)। Writes दोबारा पढ़ते हैं, merge करते हैं और atomically rename करते हैं,
इसलिए एक साथ tune कर रहे दो processes ज़्यादा से ज़्यादा एक-दूसरे की सबसे नई line खोते हैं। न पढ़ी या न लिखी
जा सकने वाली directory एक miss है, कभी error नहीं; कोई writable root न हो तो store सिर्फ़ memory में रहता है।

---

## इसे बंद करना

| Control | असर |
|---|---|
| `SVOD_TK_TUNE=0` | कोई measurement नहीं; हर policy अपना static चुनाव लौटाती है (`GemmPolicy::cfg`, `FaPolicy::config`, policy का split) |
| `svod_tk::tune::set_enabled(false)` | वही, code से, process के लिए environment को override करते हुए — test harnesses इसे call करते हैं ताकि कोई कर्नेल test हर छुए गए shape को tune न करे |
| `gemm_nt_with(x, w, cfg)`, `flash_attention_tuned(q, k, v, opts, policy)`, `SqAttentionOpts::split` | एक launch के लिए अपने chooser से policy को bypass करें |

जब device कोई dispatch timestamps stamp नहीं करता (`dispatch_gpu_ns` `None` है), तब भी tuning छोड़ दी जाती है,
क्योंकि तुलना करने को कुछ होगा ही नहीं।

---

## क्या tune होता है

| कर्नेल | Candidates | Table |
|---|---|---|
| `gemm_nt` | family की table का हर `GemmCfg` जो `(m, k, n)` को tile करता है और epilogue रखता है | `tk/src/kernels/gemm.rs` में `CUDA_TILES` (2), `RDNA_TILES` (3), `RDNA4_TILES` (5) |
| `flash_attention` | `FA_TILES` का हर `(q_blk, kv_blk)` जिसके K/V double buffers shared memory में फ़िट हों और जिसका block `N` को divide करे | `tk/src/kernels/fa.rs` में `[(16,16), (16,32), (16,64), (32,32)]` |
| `single_query_attention` | `N` के वे divisors जो device के resident-wave budget के सबसे नज़दीक हों और हर chunk को कम से कम 15 trips दें | `tk/src/kernels/sq_attention.rs` में `SqPolicy::candidates` |

हर policy जिस static चुनाव पर लौटती है, वह ख़ुद भी हर family के एक part पर measure किया गया है: GEMM का
`GemmPolicy::cfg` सबसे चौड़ी tile पसंद करता है, जब तक उसका grid device के compute units को `resident` गुना
न भर दे; `FaPolicy::tile` ऊँची per-warp tile तभी चुनता है जब launch grid device को cover कर ले और head dim
family की सीमा से नीचे हो। Tuning इसलिए है क्योंकि ये crossovers shape के साथ खिसकते हैं।

:::tip[Measurement पढ़ना]
`SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib tune::gemm_first_use -- --ignored` एक scratch store के
ख़िलाफ़ असली क्रम चलाता है और assert करता है: एक file, एक line, और यह कि दूसरी request बिना measure किए उसे
वापस पढ़ती है। अपने run में किसी shape के लिए क्या चुना गया, यह देखने के लिए file पढ़ें: index ऊपर बताई गई
table में एक position है।
:::

कीमत हर device पर हर shape के लिए एक बार चुकती है: कुछ compiles, और ठंडे GPU के लिए लगभग दो seconds की
timing। जिस model के shapes bucketed हों — Qwen3 sequence lengths को `FLASH_ATTENTION_SEQUENCE_MULTIPLE` तक
bucket करता है — वह मुट्ठी भर lines tune करता है और फिर हर batch memo से चलाता है।
