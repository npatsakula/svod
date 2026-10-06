---
sidebar_label: कर्नेल सर्च
---

# कर्नेल सर्च: ह्यूरिस्टिक्स, BEAM और टेंसर कोर

`apply_pre_optimization` के बाद कर्नेल `Weak` और `Reduce` रेंज का एक लूप नेस्ट होता है। ऑप्टिमाइज़र तय करता है कि ये लूप कैसे चलेंगे — कौन-से ग्रिड डाइमेंशन, वर्कग्रुप, वार्प, वेक्टर लेन, अनरोल किए गए बॉडी या टेंसर-कोर फ़्रैगमेंट बनेंगे — और यह `Scheduler` पर `Opt` लागू करके किया जाता है। opts चुनने की दो रणनीतियाँ हैं: हाथ से लिखे ह्यूरिस्टिक्स (डिफ़ॉल्ट) और BEAM सर्च। स्रोत: `schedule/src/optimizer/{scheduler,opts,heuristics,beam,tc,renderer,config}.rs`, `ir/src/opt.rs`।

## शेड्यूलर और एक्शन स्पेस

`Scheduler::new(ast, renderer)` कर्नेल के `RANGE` (extent > 1) को इंडेक्स करता है, `(axis_type.priority(), axis_id)` के क्रम में; `convert_loop_to_global` उन `Weak` अक्षों को, जो हर `STORE` में आते हैं, `Global` में बदल देता है जब रेंडरर `has_local` (GPU) हो, और CPU पर कुछ नहीं करता। इसके बाद `apply_opt(scheduler, opt, append)` हर कॉल में एक रेंज को फिर से लिखता है:

| `OptOps` | प्रभाव | शर्तें (`opts.rs`) |
|----------|--------|--------------------|
| `UPCAST(axis, n)` | `Global`/`Local`/`Weak` अक्ष से `n` लेन को `Upcast` के रूप में अलग करना | `n <= renderer.upcast_max`; `n = 0` पूरा अक्ष ले लेता है |
| `UNROLL(axis, n)` | `Reduce`/`GroupReduce` अक्ष से `n` इटरेशन को `Unroll` के रूप में अलग करना | `axis` `unrollable_dims()` में इंडेक्स है; `n <= 32` |
| `LOCAL(axis, n)` | `Global`/`Weak` अक्ष से वर्कग्रुप डाइमेंशन अलग करना | `has_local`, पहले कोई `NOLOCALS` नहीं |
| `GROUP(axis, n)` / `GROUPTOP(axis, n)` | `Reduce` अक्ष का भीतरी / बाहरी विभाजन `GroupReduce` में (शेयर्ड मेमोरी के ज़रिए दो-चरणीय रिडक्शन) | `has_local && has_shared`, `shared_max` में फ़िट, किसी दूसरे reduce में नेस्टेड नहीं, **TC opt लागू होने के बाद अस्वीकार** |
| `THREAD(axis, n)` | ग्लोबलाइज़ेबल `Global`/`Weak` अक्ष पर CPU कोर डाइमेंशन | `has_threads`, कोई मौजूदा `Thread` अक्ष नहीं, `n <= global_max[0]` |
| `SWAP(a, b)` | दो `Global` अक्षों की अदला-बदली | दोनों `Global` — इसलिए CPU पर कभी नहीं, जहाँ अक्ष `Weak` रहते हैं |
| `PADTO(axis, n)` | अक्ष को `n` के गुणज तक पैड करना, बचे हिस्से को मास्क करते हुए | स्थिर extent, `Upcast`/`Unroll`/`Thread` नहीं, पैडिंग काम के 4× से कम, एकल-इंडेक्स `INDEX` |
| `NOLOCALS` | `dont_use_locals` सेट करना, बाद के `LOCAL` रोकना; gpudims सिर्फ़ global लॉन्च करता है | अभी तक कोई `Local`/`Warp`/`GroupReduce` अक्ष नहीं |
| `TC(axis_choice, tc_select, tc_opt, use_tc)` | मैटमल को टेंसर कोर पर मैप करना | पहला opt होना चाहिए; नीचे देखें |

`get_optimized_ast_with_naming` रेंज सूचियों को फ़्लैट करता है और `KernelInfo { name, applied_opts, dont_use_locals }` जोड़ता है; नाम `r_`/`E_` के बाद रेंज क्रम में extents होता है ([वर्क्ड उदाहरण](../codegen/worked-example.md) में `r_8_16_4`)।

## ह्यूरिस्टिक्स (`hand_coded_optimizations`)

`hand_coded_optimizations(&mut scheduler, &HeuristicsConfig)` इस क्रम में लागू करता है (`heuristics.rs`):

1. **`try_tensor_cores`** — यदि `tc_enabled != Disabled` हो, रेंडरर में कोर हों और (`TcOpt::Strict` के तहत) ठीक एक reduce अक्ष हो: `tc::detect_matmul`, फिर अक्ष विकल्पों पर `apply_with_axis_choice`, फिर `apply_tc_tiling` — `FixedStep`: M और N का `UPCAST` 5/4/3/2 में से पहले भाजक से, N का `LOCAL` 4 या 2 से; `LaneBudget { accum_max: 128 }` (CUDA sm75/80/89): `tc_warp_tile_growth` फिर `wave_size / tc.threads` का `LOCAL`। सफल होने पर लौट जाता है।
2. **`apply_image_upcasts`** — इमेज बफ़र।
3. **`apply_matvec_fast_path`** — `SVOD_MV*` matvec कॉन्फ़िगरेशन (`PADTO`, छोटे अक्षों का `UPCAST`, best-effort `GROUP`, `LOCAL`, `UPCAST`, `UNROLL`)। सफल होने पर लौट जाता है।
4. **`try_grouped_reduction`** — अधिकतम 2048 एलिमेंट वाले आउटपुट के लिए `GROUPTOP(axis, 16)` (locals के बिना 240); अन्यथा **`try_warp_row_reduction`** (वेव साइज़ से `GROUP` और `UNROLL 4`)। यदि अब कोई `GroupReduce` अक्ष मौजूद है तो फ़ंक्शन लौट जाता है।
5. **`apply_masked_upcasts`** — 2–7 आकार के मास्क्ड अक्ष, जिनका गुणनफल ≤ 49 हो।
6. **`apply_heuristic_upcasts`** — जब तक आउटपुट में ≥ 1024 एलिमेंट हों और upcast गुणनफल 32 से कम हो, 3 या 4 से `UPCAST`, अक्ष `(num_strides, sum_strides, axis, vector rank)` के अनुसार रैंक किए जाते हैं।
7. **`apply_unroll`** — reduce अक्ष ≤ 32 होने पर पूरी तरह अनरोल होता है (दोनों ≤ 3 हों तो दूसरा भी), अन्यथा `UNROLL 4`।
8. **`apply_default_upcast`** — यदि अभी तक कुछ भी upcast या unroll नहीं हुआ, तो आख़िरी upcastable अक्ष पर `UPCAST 4`।
9. **`apply_local_dims`** — अक्ष 0 के लिए `LOCAL` आकार `[32, 16, 8, 4, 3, 2]` और बाकी के लिए `[16, 8, 4, 3, 2]`, संचयी बजट 128, अधिकतम तीन, `PADTO` फ़ॉलबैक के साथ।
10. **`apply_threading`** — सिर्फ़ CPU: `Weak` अक्ष पर `[32, 16, 12, 8, 6, 5, 4, 3, 2]` से `THREAD`, प्रति थ्रेड कम से कम 131072 एलिमेंट रखते हुए, `PADTO` + `THREAD` फ़ॉलबैक के साथ।

`HeuristicsConfig::from_env` पढ़ता है `SVOD_TC` (0 बंद, 2 सिर्फ़ आकार, अन्यथा चालू), `SVOD_TC_OPT`/`TC_OPT`, `SVOD_TC_SELECT`/`TC_SELECT`, `SVOD_MV*`, `SVOD_NOLOCALS`, `SVOD_THREADS`। grouped-reduction की सीमाएँ `heuristics.rs` में स्थिरांक हैं; `SVOD_K_VECTORIZE` और `SVOD_NO_OUTPUT_UPCAST` ऐसे फ़ील्ड सेट करते हैं जिन्हें इस पथ पर कोई नहीं पढ़ता।

## BEAM सर्च

`BEAM=N` (N > 0) `OptStrategy::Beam { width: N }` चुनता है। तब `realize` कर्नेल को `beam_search_cached_remote(scheduler, config, compiler_identity, behavior_fingerprint, compile_wave, benchmark)` (`beam.rs`) से गुज़ारता है; सादे `optimize_kernel_with_config` API में compile-और-time क्लोज़र नहीं होता, इसलिए वह ह्यूरिस्टिक्स पर लौट आता है।

सर्च (`beam_search_remote_staged`):

1. `[(scheduler, Duration::MAX)]` से शुरू करें और ह्यूरिस्टिक्स के परिणाम को पहली वेव के अतिरिक्त उम्मीदवार के रूप में जोड़ें।
2. **विस्तार**: हर beam सदस्य के लिए `generate_actions` 193 `BEAM_ACTIONS` में से हर एक आज़माता है (`BEAM_PADTO` के साथ 200): `passes_prefilter` (अक्ष मौजूद है; जिस action की मात्रा अक्ष के आकार के बराबर हो, उसे छोड़ दिया जाता है जब `0` वैरिएंट मौजूद हो), `apply_opt`, `validate_limits` (`upcast_prod / tc_up <= max_upcast`, `local_prod <= max_local`)। `enable_nolocals` होने पर हर सदस्य के लिए `NOLOCALS` जोड़ा जाता है।
3. **कंपाइल**: उम्मीदवारों को वर्कर प्रोसेस के पूल में कंपाइल किया जाता है; यदि लीनियराइज़्ड op गिनती `max_uops` तक पहुँचे या कंपाइलेशन `compile_timeout_secs` से अधिक हो, तो उम्मीदवार वहीं हटा दिया जाता है।
4. **फ़िल्टर**: जिन उम्मीदवारों के `compute_ops` वेव के न्यूनतम से 1000× से अधिक हों, वे हटाए जाते हैं, फिर बाइनरी (या सोर्स) कुंजी के अनुसार डुप्लिकेट।
5. **समय मापन**: हर एक के `num_runs` रन, स्कोर = न्यूनतम; कोई रन मौजूदा सर्वश्रेष्ठ के 3× पर रोक दिया जाता है; global आकार 65536 पर सीमित किया जाता है और समय वापस स्केल किया जाता है।
6. **रखना**: सर्वश्रेष्ठ `beam_width` रखे जाते हैं। जब सर्वश्रेष्ठ समय `min_progress_ns` से अधिक नहीं सुधरता (या पहले से उससे कम है), तब रुकें; सुधार होने पर अगली वेव के लिए beam एक विजेता तक सिमट जाता है।

एक्शन सूची (`BEAM_ACTIONS`): `UPCAST` मात्राएँ `[0,2,3,4,5,7]` × अक्ष 0..8 (48), `UNROLL` `[0,4,7]` × 0..5 (15), `LOCAL` `[2,3,4,8,13,16,29]` × 0..6 (42) तथा `(0,32)` और `(6,2)`, `GROUPTOP` `[13,16,28,29,32,49,64,256]` × 0..3 (24), `GROUP` `[0,4,8,16]` × 0..3 (12), `TC` (एक `tc_opt = 0` action और `TC_OPT` पर नौ अक्ष विकल्प), 0..5 के भीतर `SWAP` जोड़े (10), `THREAD` `[2,3,4,5,8,12,16,24,32,64]` × 0..3 (30)। `BEAM_PADTO` अक्ष 0..7 के लिए `PADTO(axis, 32)` जोड़ता है।

### कैश

परिणाम `$SVOD_BEAM_CACHE_DIR/beam_cache` पर, अन्यथा `~/.cache/svod/beam_cache` (`dirs::cache_dir()`) पर, एक `sled` डेटाबेस में सहेजे जाते हैं। कुंजी (`CacheKey`, स्कीमा 11) है: संरचनात्मक AST हैश तथा beam चौड़ाई, डिवाइस, `renderer.cache_fingerprint()`, कंपाइलर पहचान, सीमाएँ (`max_upcast`, `max_local`, `max_uops`, `num_runs`, `min_progress_ns`, `enable_nolocals`, `compile_timeout_secs`), व्यवहार फ़िंगरप्रिंट (`transcendental`, `disable_fast_idiv`) और एक्शन स्पेस का हैश। मान `applied_opts` सूची है; हिट को `replay_opts` से दोबारा चलाया जाता है, एक बार वैलिडेट और बेंचमार्क किया जाता है, और विफल होने पर अमान्य कर दिया जाता है। `IGNORE_BEAM_CACHE=1` इसे बायपास करता है, `clear_cache` इसे खाली करता है।

### एनवायरनमेंट

| वेरिएबल | डिफ़ॉल्ट | अर्थ |
|----------|---------|---------|
| `BEAM` | 0 | beam चौड़ाई; 0 = ह्यूरिस्टिक्स |
| `BEAM_UPCAST_MAX`, `BEAM_LOCAL_MAX`, `BEAM_UOPS_MAX` | 256, 1024, 3000 | `validate_limits` और वर्कर op सीमा |
| `BEAM_RUNS` | 3 | प्रति उम्मीदवार टाइमिंग रन |
| `BEAM_MIN_PROGRESS` | 10 (µs, ns के रूप में संग्रहीत) | रुकने की सीमा |
| `BEAM_PADTO` | 0 | सात `PADTO` actions जोड़ना |
| `NOLOCALS` / `SVOD_NOLOCALS` | सेट नहीं | `NOLOCALS` action जोड़ना |
| `PARALLEL` | 0 | कंपाइल वर्कर (GPU पर डिफ़ॉल्ट थ्रेड बजट, अन्यथा 1) |
| `BEAM_TIMEOUT_SEC`, `BEAM_MAX_TASKS_PER_CHILD` | 10, 16 | वर्कर वॉचडॉग और रीसाइक्लिंग |
| `TC`, `TC_OPT` | 1, 2 | BEAM के टेंसर-कोर actions (BEAM के तहत `TC_SELECT` अनदेखा होता है: हमेशा `Auto`) |
| `BEAM_DEBUG`, `BEAM_LOG_SURPASS_MAX` | सेट नहीं | डायग्नॉस्टिक्स |
| `IGNORE_BEAM_CACHE`, `SVOD_BEAM_CACHE_DIR` | सेट नहीं | कैश नियंत्रण |

:::tip[BEAM ह्यूरिस्टिक्स स्विच नहीं पढ़ता]
BEAM के भीतर ह्यूरिस्टिक सीड `HeuristicsConfig::from_env()` का उपयोग करता है, लेकिन सर्च के अपने TC actions `TC` और `TC_OPT` पढ़ते हैं, `SVOD_TC`/`SVOD_TC_OPT` नहीं। `SVOD_NOOPT` (कोई भी मान) `OptStrategy::None` चुनता है: कोई opts नहीं, लेकिन pre- और post-optimization फिर भी चलते हैं।
:::

## टेंसर कोर

`renderer.rs` हर `RendererDevice` के लिए कोर तालिका रखता है; dims `(N, M, K)` हैं:

| लक्ष्य | कोर (इनपुट → आउटपुट) | threads |
|--------|------------------|---------|
| CUDA sm75 | 8×16×8 f16→f32, f16→f16 | 32 |
| CUDA sm80 | 8×16×16 f16→f32, bf16→f32, f16→f16; 8×16×8 f16→f32, f16→f16; 8×16×32 i8→i32; वैकल्पिक tf32 8×16×8 | 32 |
| CUDA sm89 | sm80 तथा 8×16×32 fp8 e4m3/e5m2→f32 | 32 |
| AMD RDNA3 | 16×16×16 f16→f32, f16→f16, bf16→f32, i8→i32 | 32 |
| AMD RDNA4 | RDNA3 तथा bf16→bf16 | 32 |
| AMD CDNA3 | 16×16×32 fp8 e5m2/e4m3; 16×16×16 f16/bf16→f32 | 64 |
| AMD CDNA4 | CDNA3 तथा 16×16×128 fp8 | 64 |
| Metal | 8×8×8 f32/f16/bf16 वैरिएंट | 32 |
| Intel Xe | 8×8×16 f16→f32 | 8 |
| WebGPU, CPU | कोई नहीं | — |

`for_cuda_arch` sm80 प्रोफ़ाइल चुनता है जब क्षमता में bf16 mma हो, अन्यथा sm75, और sm75 से नीचे कोई कोर नहीं।

`tc.rs`: `detect_matmul` `REDUCE(Add, MUL(in0, in1), reduce_ranges)` ढूँढता है; जो रेंज सिर्फ़ `in0` उपयोग करता है वे M उम्मीदवार हैं, जो सिर्फ़ `in1` उपयोग करता है वे N, reduce रेंज K हैं, और हर `(M, N, K)` त्रिक एक अक्ष विकल्प है (ऐसी M/N रेंज जो स्वयं `Reduce` अक्ष हो, अस्वीकार की जाती है); `select_tensor_core` इनपुट और आउटपुट स्केलर dtypes का मिलान करता है (नेटिव कोर के बिना fp8 इनपुट f16 कोर पर लौट आता है); `apply_with_axis_choice` 64 प्रयासों के बजट में अक्ष विकल्पों × कोर पर लूप करता है। कोर अक्षों को विभाजित करके लागू किया जाता है: extent `tc.threads` की एक `Warp` रेंज, हर `TcOpt::Upcast` प्रविष्टि के लिए आकार 2 का एक `Upcast` अक्ष, हर `TcOpt::Local` प्रविष्टि warp इंडेक्स का एक अंक लेती है (`warp % 2`, `warp / 2`), K आकार 2 के `log2(K)` `Unroll` अक्ष बन जाता है; बचे हुए N/M `Global` रहते हैं और बचे हुए reduce अक्ष `WMMA` को एक `REDUCE` में लपेटते हैं। `TcUsage::ShapeOnly` (`SVOD_TC=2`) विभाजन करता है लेकिन कोई `WMMA` नहीं बनाता।

`TcOpt` स्तर (`TC_OPT`): **0 Strict** — सिर्फ़ एक reduce अक्ष, M/N/K विभाज्य होने चाहिए; **1 Relaxed** — `tc.rs` के भीतर वही विभाज्यता नियम; **2 Padded** (डिफ़ॉल्ट) — अविभाज्य अक्ष का `PADTO` जब पैडिंग अधिकतम 25% जोड़े; **3 Unbounded** — `PADTO` की अपनी 4× सीमा के भीतर पैड करना। सिम्बॉलिक अक्ष कभी टेंसर कोर का उपयोग नहीं करता।

## प्रोग्रामेटिक कॉन्फ़िगरेशन

```rust
use svod_schedule::optimizer::{OptStrategy, OptimizerConfig};
use svod_tensor::PrepareConfig;

let config = PrepareConfig::from(
    OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 8 })
        .build(),
);
tensor.realize_with(&config)?;
```

`OptimizerConfig` (`bon` बिल्डर) में `strategy`, `beam: BeamConfig`, `heuristics: HeuristicsConfig`, `transcendental` (`TRANSCENDENTAL`, डिफ़ॉल्ट 1; ≥ 2 बहुपद विघटन को बाध्य करता है), `disable_fast_idiv` (`DISABLE_FAST_IDIV`, डिफ़ॉल्ट **1**: मैजिक-नंबर डिवीज़न `DISABLE_FAST_IDIV=0` से ऑप्ट-इन है) और `opts_to_apply` (opts की स्पष्ट सूची, जो कर्नेल `SINK` के `KernelInfo` से भी पढ़ी जा सकती है; यह रणनीति को ओवरराइड करती है और लागू न हो पाने वाला opt एक त्रुटि है) होते हैं। `PrepareConfig` में `optimizer`, `planner_mode`, `disable_schedule_cache`, `device_local_outputs`, `threads` और कंस्ट्रक्टर `Default`, `from_env`, `device_local`, `for_cpu_backend`, `for_{amd,metal,cuda}_if_available` होते हैं।
