---
sidebar_label: Tuning
---

# Tuning

हर op के पास `ops::config` में हर shape के लिए एक candidate list है (tables के लिए
[कर्नेल लाइब्रेरी](./kernel-library) देखें)। पहला candidate वह है जो untuned चलता है। जब कोई device
पहली बार किसी shape से मिलता है, tune store पूरी list मापता है और विजेता को रखता है।

बेहतर fixed default काफ़ी नहीं था। Nemotron के GEMMs (M = 704, N 512–2048) पुरानी fixed tile ladder
के साथ लगभग 14 TFLOP/s पर चलते थे: 28 SMs पर 48 blocks। Tune store के साथ RTX 3060 पर उनका औसत समय
76.5 से गिरकर लगभग 60 µs हो गया।

## मापन कब होता है {#when-measurement-happens}

Op वहीं मापता है जहाँ उसे बुलाया जाता है, ग्राफ़ बनते समय, कभी किसी चलते हुए plan के अंदर नहीं। हर
candidate एक program के रूप में बनता है, capacity पर scratch buffers पर launch होता है, हर runtime
variable उसकी अधिकतम value से bound होता है, और समय मापा जाता है:

| क़दम | सेटिंग |
|---|---|
| Warm-up | 500 ms के लगातार runs (RTX 3060 idle में 210 MHz पर रहता है) |
| Rounds | सभी candidates पर 4 round-robin rounds |
| हर round | 10 ms के sustain runs, फिर 5 profiled runs, हर candidate का न्यूनतम रखते हुए |
| एक run का समय | सबसे लंबे कर्नेल के GPU timestamps |

एक candidate programs की list है, split attention के लिए एक कर्नेल और उसके partial results का merge,
और उसका समय उसके programs के समयों का योग है। जो candidate build या run नहीं हो पाता, उसे छोड़ दिया
जाता है। अगर कुछ भी नहीं मपता, तो पहला candidate
इस्तेमाल होता है और कुछ store नहीं होता।

## Store {#the-store}

| क्या | मान |
|---|---|
| Directory | `$SVOD_TK3_TUNE_DIR`, वरना `$XDG_CACHE_HOME/svod/tk3_tune`, वरना `~/.cache/svod/tk3_tune` |
| फ़ाइल | हर device और crate version के लिए एक, जैसे `sm_86_28sm-v0.2.0.txt` |
| Line | `op\|device\|dtype\|shape\|candidates\|programs index ns` |
| Key | `tune::TuneKey { op, device, dtype, shape, candidates }`, जहाँ `device` arch और SM count है (`sm_86-28sm`) |

एक असली store की line: attention, bf16, shape `[batch, t, tk, heads, kv_heads, d]`, candidate
2 61.4 µs पर जीता।

```text
attention|sm_86-28sm|BFloat16|1x704x704x8x8x64|3733e8921b15aa79|77aee9d5c98fa463 2 61440
```

`programs` field बने हुए candidate programs और उनकी lowerings का fingerprint है, इसलिए कर्नेल बदलने पर
फिर से मापा जाता है। Writes फ़ाइल को फिर से पढ़ते हैं, merge करते हैं और atomically बदलते हैं। न पढ़ा
जा सकने वाला या न लिखा जा सकने वाला store miss गिना जाता है, कभी error नहीं। एक process memo दोहराई
गई कॉल्स का जवाब बिना कुछ बनाए देता है।

## इसे बंद करना {#switching-it-off}

| कैसे | असर |
|---|---|
| `SVOD_TK3_TUNE=0` | मापन बंद; पहला candidate चलता है |
| `svod_tk3::tune::set_enabled(false)` | इस process के लिए वही, environment को override करते हुए (tests इसे इस्तेमाल करते हैं) |

`tune::TuneStore::at(root)` किसी दूसरे root पर store बनाता है, या `None` के साथ सिर्फ़ memory में।
`tune::measure(candidates)` `tune::Candidate`
(`Vec<(Program, Lowering)>`) की किसी भी list को उसी तरह मापता है।

## Probes {#probes}

Probes `#[ignore]` tests हैं जो timings छापते हैं और कभी assert नहीं करते। इन्हें idle GPU पर एक-एक
करके चलाएँ:

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk3 --lib --release -- --ignored --nocapture --test-threads=1 gemm_candidates_probe
```

| Probe | क्या छापता है |
|---|---|
| `gemm_throughput_probe` | 4096³ पर tk1 के मुक़ाबले tk3 GEMM configs, TFLOP/s |
| `gemm_candidates_probe` | Nemotron के projection shapes और 4096³ पर हर GEMM candidate, untuned चुनाव और विजेता |
| `attention_throughput_probe` | tk1 के मुक़ाबले flash attention (B 4, H 8, T 2048; d 64/128, causal और non-causal) |
| `decode_throughput_probe` | tk1 के मुक़ाबले Whisper large-v3 decoder step का self और cross attention |
| `first_execution_probe` | Graph GEMM के मुक़ाबले tk3 GEMM को बनाने, lower करने और prepare करने की host लागत |

आख़िरी probe ने वे host लागतें मापीं जिन्होंने `launch.rs` को आकार दिया। एक GEMM body को lower करने
में 2.3 ms लगते हैं, इसलिए lowered bodies program, lowering, device और placeholder shapes के हिसाब से
memoize होती हैं (hit पर 30 µs)। Prepare के समय एक body को schedule करना अब भी graph GEMM से लगभग
0.35 ms ज़्यादा लेता है।
