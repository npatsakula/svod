---
sidebar_label: Tuning
---

# Tuning

हर op के पास `ops::config` में हर shape के लिए एक ranked config list है (tables के लिए
[कर्नेल लाइब्रेरी](./kernel-library) देखें), जो device की मापी गई costs से rank होती है
(देखें [Device costs](#device-costs))। Ranking एक model है, और register count भी, इसलिए दोनों में से
कोई config admit नहीं करता। एक नियम करता है: tuner ranked configs को आठ-आठ के batches में, ranked क्रम में
और parallel में, तब तक compile करता है जब तक उसके पास वे पहले आठ tiles न हों जिनके compiled kernels spill
नहीं करते (कोई kernel spill किए गए scratch bytes बताए तो spill; जो backend कुछ नहीं बताता वह admit करता है)।
Ranking किसी tile की हर pipeline (ring depth, unrolling) सूचीबद्ध करती है, और tile अपनी पहली clean pipeline
पर admit होता है, इसलिए spill करने वाली unrolled ring अपने rolled जुड़वाँ के लिए जगह छोड़ती है।
Spilling किसी config को तभी तक बाहर रखती है जब तक clean configs बचे हों: जब हर compile होने वाला config
spill करे, तो उनमें से पहले आठ फिर भी मापे जाते हैं, क्योंकि फ़ैसला समय करता है और spill करने वाला kernel भी
ग्राफ़ को हरा देता है। एक config की list भी इसी रास्ते से जाती है, और कोई op spills की वजह से ग्राफ़ पर नहीं
गिरता। Admit हुए configs, अपने variants के साथ (attention के लिए key splits, भूखी convolution के लिए
reduction splits), वे candidates हैं जिन्हें tune store तब मापता है जब कोई device पहली बार किसी shape से
मिलता है, और वह विजेता को रखता है। Attention अपने softmax forms और अपने tiles बारी-बारी tune करता है: ranking
के पहले clean config पर forms, उन forms के साथ tiles, फिर winner पर दोबारा forms (देखें
[कर्नेल लाइब्रेरी](./kernel-library))। Admit हुई list सिर्फ़ process के लिए याद रखी जाती है; दूसरा process उसे
फिर compile करता है, जिसे compiler का object cache सस्ता बनाता है। जब कुछ भी compile नहीं होता (compile करने
को कोई device नहीं) तो पहले आठ बिना परखे इस्तेमाल होते हैं।

Tuning बंद होने पर (`SVOD_TK3_TUNE=0`) admission फिर भी compile करता है और पहला admit हुआ config चलता है;
सिर्फ़ candidate timing बंद होती है। Device के cost probes फिर भी मापते हैं, हर device पर एक बार
(gfx1151 पर लगभग 4 s), क्योंकि ranking को उनकी ज़रूरत है।

बेहतर fixed default काफ़ी नहीं था। Nemotron के GEMMs (M = 704, N 512–2048) पुरानी fixed tile ladder
के साथ लगभग 14 TFLOP/s पर चलते थे: 28 SMs पर 48 blocks। Tune store के साथ RTX 3060 पर उनका औसत समय
76.5 से गिरकर लगभग 60 µs हो गया।

## मापन कब होता है {#when-measurement-happens}

Op वहीं मापता है जहाँ उसे बुलाया जाता है, ग्राफ़ बनते समय, कभी किसी चलते हुए plan के अंदर नहीं। हर
candidate एक program के रूप में बनता है, capacity पर scratch buffers पर launch होता है, हर runtime
variable उसकी अधिकतम value से bound होता है, और समय मापा जाता है:

| क़दम | सेटिंग |
|---|---|
| Warm-up | लगातार runs, जब तक clock स्थिर न हो जाए (RTX 3060 idle में 210 MHz पर रहता है) |
| Rounds | सभी candidates पर 4 round-robin rounds |
| हर round | 10 ms के sustain runs, फिर 5 profiled runs, हर candidate का न्यूनतम रखते हुए |
| एक run का समय | उसके कर्नेलों के GPU timestamps का योग |

एक candidate programs की list है, split attention के लिए एक कर्नेल और उसके partial results का merge,
और उसका समय उसके programs के समयों का योग है। जो candidate build या run नहीं हो पाता, उसे छोड़ दिया
जाता है। अगर कुछ भी नहीं मपता, तो पहला candidate इस्तेमाल होता है और कुछ store नहीं होता।

दो guards समय को ईमानदार रखते हैं:

- **एक device lock।** Process की हर timing, tuner की और cost probes की भी, `tune::time` के दौरान एक ही
  lock रखती है: device साझा करने वाले दो माप एक-दूसरे को माप लेते। एक ही key tune करने वाले दो threads उसे
  एक बार मापते हैं; दूसरा पहले के जवाब का इंतज़ार करता है।
- **Stamp guard।** gfx1151 कभी-कभी किसी लंबे kernel पर लगभग एक microsecond का stamp लगा देता है (90 ms के
  kernel के लगभग साठ runs में एक बार देखा गया)। `tune::best` सबसे तेज़ रखने से पहले उस run को हटा देता है जो
  उसके candidate के runs के median के आधे से कम हो, tuner और probes दोनों के लिए।

## Store {#the-store}

| क्या | मान |
|---|---|
| Directory | `$SVOD_TK3_TUNE_DIR`, वरना `$XDG_CACHE_HOME/svod/tk3_tune`, वरना `~/.cache/svod/tk3_tune` |
| फ़ाइल | हर device और crate version के लिए एक, जैसे `sm_86_28sm-v0.2.0.txt` |
| Line | `op\|device\|dtype\|shape\|candidates\|compiler\|programs index ns` |
| Key | `tune::TuneKey { op, device, dtype, shape, candidates, compiler }`, जहाँ `device` arch और SM count है (`sm_86-28sm`) और `compiler` device के compiler का digest है (`tune::compiler`) |

एक असली store की line: attention, f16, shape `[batch, t, tk, heads, kv_heads, d]`, candidate
1 23.2 µs पर जीता।

```text
attention|gfx1151-40sm|Float16|5x1x448x20x20x64|f0896a3761c96a19|465b6912a7cb2da1|9c2caed511ea3c95 1 23240
```

`programs` field candidates के programs और lowerings का fingerprint है, और `compiler` toolchain का, इसलिए
कर्नेल या compiler बदलने पर फिर से मापा जाता है। Fingerprint के लिए programs बनाए जाते हैं (सस्ता); सिर्फ़
store miss उन्हें compile करता है, इसलिए warm store admission के आगे कुछ compile नहीं करता। Writes फ़ाइल को फिर से पढ़ते हैं, merge करते हैं और atomically बदलते हैं। न पढ़ा
जा सकने वाला या न लिखा जा सकने वाला store miss गिना जाता है, कभी error नहीं। एक process memo दोहराई
गई कॉल्स का जवाब बिना कुछ बनाए देता है।

## Device costs {#device-costs}

Ranking की costs (`atoms::Costs`: एक shared byte, एक streamed byte, एक matrix-core step, एक score का
softmax, एक pipeline trip, lane moves, global latency, ...) device पर `probe::costs` से मापी जाती हैं, पहली
बार जब कोई process उसके लिए plan करता है: हर field programs की एक जोड़ी है, एक probe जो वह सब करता है जो
उसका baseline करता है, साथ में primitive की एक ज्ञात मात्रा। Values उसी store फ़ाइल में `probe:<field>` lines
के रूप में जाती हैं (index 0, `ns` slot में femtoseconds), ताकि बाद के processes उन्हें वापस पढ़ें। Planner हर
device पर एक बार बनता है; दूसरा caller probes का इंतज़ार करता है। जिस device की SM count नहीं है वह किसी probe
को scale नहीं कर सकता और उसका कोई planner नहीं होता। Host tests इसकी जगह costs declare करते हैं:
`test::unit::costs()` और `gfx1151_costs()`, बाद वाला एक gfx1151 store से चिपकाया गया।

## इसे बंद करना {#switching-it-off}

| कैसे | असर |
|---|---|
| `SVOD_TK3_TUNE=0` | Candidate timing बंद; पहला admit हुआ candidate चलता है (cost probes फिर भी मापते हैं) |
| `svod_tk3::tune::set_enabled(false)` | इस process के लिए वही, environment को override करते हुए (tests इसे इस्तेमाल करते हैं) |

`tune::TuneStore::at(root)` किसी दूसरे root पर store बनाता है, या `None` के साथ सिर्फ़ memory में।
`tune::measure(candidates)` `tune::Candidate`
(`Vec<(Program, Lowering)>`) की किसी भी list को उसी तरह मापता है।

## Probe tests {#probe-tests}

Probe tests `#[ignore]` tests हैं जो timings छापते हैं और कभी assert नहीं करते। वे device की probe की गई
costs से rank करते हैं और उस shortlist को मापते हैं जिसे tuner admit करता है (op layer का अपना admission),
tuner के चुनाव को चिह्नित करते हुए। इन्हें idle GPU पर एक-एक करके चलाएँ:

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk3 --lib --release -- --ignored --nocapture --test-threads=1 gemm_candidates_probe
```

| Probe | क्या छापता है |
|---|---|
| `gemm_throughput_probe` | 4096³ पर admit हुए GEMM configs, TFLOP/s, और tuner का चुनाव |
| `gemm_candidates_probe` | Nemotron के projection shapes और 4096³ पर हर admit हुआ GEMM candidate, untuned चुनाव, सबसे तेज़ और tuner का चुनाव |
| `attention_throughput_probe` | Flash attention (B 4, H 8, T 2048; d 64/128, causal और non-causal): हर admit हुआ config और tuner का चुनाव उसके softmax forms के साथ |
| `decode_throughput_probe` | Whisper large-v3 decoder step का self और cross attention |
| `first_execution_probe` | Graph GEMM के मुक़ाबले tk3 GEMM को बनाने, lower करने और prepare करने की host लागत |

आख़िरी probe ने वे host लागतें मापीं जिन्होंने `launch.rs` को आकार दिया। एक GEMM body को lower करने
में 2.3 ms लगते हैं, इसलिए lowered bodies program, lowering, device और placeholder shapes के हिसाब से
memoize होती हैं (hit पर 30 µs)। Prepare के समय एक body को schedule करना अब भी graph GEMM से लगभग
0.35 ms ज़्यादा लेता है।
