---
sidebar_label: Tuning
---

# Tuning {#tuning}

हर op `ops::config` में अपने configs को हर shape के लिए device की मापी गई costs से rank करता है (देखें
[Device costs](#device-costs)), और 16 tiles रख लेने के बाद सबसे सस्ते के 2.5× से महँगे configs छोड़ देता है
(`config::PRUNED_PAST`): sm_86 पर उनमें से कोई contender band में नहीं चला। Ranking एक model है, और register count भी, इसलिए दोनों में से कोई
config को admit नहीं करता। एक नियम करता है: tuner ranked configs को समानांतर रूप से compile करता है, जब
तक उसे ranked क्रम में वे पहले 64 tiles (`config::SHORTLIST`) न मिल जाएँ जिनके compiled kernels spill
नहीं करते (spilled scratch bytes बताने वाला कर्नेल spill करता है; जो backend कुछ नहीं बताता वह admit
करता है)। एक tile अपनी पहली clean pipeline (ring depth, unrolling) पर admit होता है, इसलिए spill करने
वाली unrolled ring अपने rolled जुड़वाँ के लिए जगह छोड़ देती है। कोई pipeline तभी compile होती है जब
उसके tile की उससे ऊपर ranked pipelines spill या fail हुईं, और 64वें clean tile के बाद कोई नहीं: ठीक
वही जो ranked क्रम में एक-एक config compile करने वाला walk compile करता (`config::Search`)। जिन tiles की कोई
pipeline clean compile नहीं होती, वे उन्हीं 64 के भीतर clean tiles के बाद आते हैं, हर एक की एक pipeline:
compiled kernel जो आँक सकता है वह आँकता है और बाक़ी समय आँकता है (spills की वजह से कोई op ग्राफ़ पर नहीं
गिरता, और gfx1201 पर 16384² depth-768 nearest का सबसे अच्छा tile प्रति lane 196 bytes spill करके clean
pick से 2 % आगे चलता है)। किसी config का compile thread pool पर तभी शुरू हो जाता है जब search उसे माँगती है (pool में जगह हो तो tile की अगली pipeline तभी शुरू हो जाती है जब पिछली अभी compile हो रही हो, और उसका verdict वहीं गिना जाता है जहाँ walk उसे माँगता); जिसका compiler
30 s (`tune::COMPILE_TIMEOUT`) से ज़्यादा चले उसे kill करके fail की तरह reject किया जाता है, और planner
log यह बताता है (`compile timed out`); जिसका lowered program उसके target की अनुमति से ज़्यादा instructions
emit करे (`Target::most_instructions`: AMD पर 2^18, CUDA पर 2^19, सबसे बड़े ranked kernels से ऊपर) उसे
lowering ही किसी compiler के देखने से पहले मना कर देती है (`lower::Error::TooLarge`, planner log में
`does not lower`) और वैसे ही reject किया जाता है। हर program एक tuning में एक बार lower होता है: store के fingerprint
ने जो lower किया वही body compile launch करता है।

Admit हुए configs, अपने variants के साथ (attention के लिए key splits, भूखे convolution के लिए
reduction splits), वे candidates हैं जिन्हें tune store तब मापता है जब कोई device पहली बार किसी shape
से मिलता है। फिर अग्रणी tiles की बाक़ी pipelines, जो अग्रणी के contender band के भीतर हों और अधिकतम आठ
(`config::TWIN_TILES`), compile होकर विजेता के ख़िलाफ़ मापी जाती हैं, spill करें या न करें, और सबसे तेज़
चुनी जाती है। Attention अपने softmax forms और अपने tiles को बारी-बारी से tune करता है (देखें
[कर्नेल लाइब्रेरी](./kernel-library#flash-attention))। जब कुछ भी compile नहीं होता (compile करने के लिए
कोई device नहीं), तो पहले 64 tiles बिना जाँचे इस्तेमाल होते हैं।

Tuning बंद होने पर कुछ भी मापा नहीं जाता: ranked list का पहला config जो clean compile हो वह चलता है,
जिसे आठ-आठ के ranked batches में compile करके ढूँढा जाता है और store में एक `first` line के रूप में रखा
जाता है। Device के cost probes फिर भी मापते हैं, क्योंकि ranking को उनकी ज़रूरत है।

## मापन कब होता है {#when-measurement-happens}

Op वहीं मापता है जहाँ उसे बुलाया जाता है, ग्राफ़ बनते समय, कभी किसी चलते plan के भीतर नहीं। हर
candidate एक program के रूप में बनता है, capacity वाले scratch buffers पर launch होता है जहाँ हर runtime
variable अपने maximum पर bound होता है (float operands में `[-1, 1)` से लिए गए मान होते हैं, ताकि जो
कार्ड असली data पर clock घटाता है वह वैसे ही मापा जाए जैसे वह चलेगा; integer operands में शून्य होते
हैं), और मापा जाता है:

| क़दम | सेटिंग |
|---|---|
| Warm-up | लगातार runs जब तक clock स्थिर न हो जाए |
| Rounds | ranked क्रम में हर candidate का एक round, जिसमें जिस candidate का पहला timed run अब तक के सबसे अच्छे के 2× से ज़्यादा हो, उसका वही run उसका time रहता है; फिर contenders के 3 और: वे जो पहले round के सबसे अच्छे के 1.2× के भीतर हों (8 से कम candidates हों तो 2× के भीतर), दूसरे round के बाद सिर्फ़ वे जो अब तक के सबसे अच्छे के 1.05× के भीतर हों, और बाद के rounds बारी-बारी से उल्टे और सीधे क्रम में चलते हैं ताकि हर candidate अपने दोनों पड़ोसियों के बाद चले; फिर दो अग्रणियों के 4 और, जहाँ दो से ज़्यादा contenders चले हों। tuning की पहली timing, जो सिर्फ़ अग्रणी tiles चुनती है, दूसरे round के बाद रुक जाती है: उन tiles को उनकी बाकी pipelines के साथ फिर से मापा जाता है |
| हर round | candidate के programs के क्रम से लगातार sustain runs में 5 ms device time (अधिकतम 1024 launches, candidate के पिछले run के हिसाब से आकार), फिर हर program के profiled runs, हर candidate का minimum रखते हुए: जितने 20 ms device time में आएँ, 2 से 5 (`tune::runs`), और 5 ms या उससे लंबे candidate के लिए कोई sustain नहीं, जिसका पहला timed run तब उन्हीं में गिना जाता है; leaders के rounds (और cost probes) sustain के बाद 5 runs रखते हैं |
| एक run का समय | उसके कर्नेलों के GPU timestamps का जोड़ |

Sustain का आकार device time में तय होता है क्योंकि launches asynchronously queue होते हैं। एक
candidate programs की एक list है, एक कर्नेल और split होने पर उसके partial results का merge, और उसका
समय उसके programs के समयों का जोड़ है। जो candidates timing शुरू होने तक compile नहीं हुए (split के variants,
tile के twins) वे pool के एक को छोड़ बाकी सभी threads पर compile होते हैं जबकि तैयार candidates मापे जाते हैं,
हर एक तब जब ranked क्रम उस तक पहुँचे। जो candidate build या run होने में विफल हो उसे छोड़ दिया जाता
है। अगर कुछ भी नहीं मापा जाता, तो पहला candidate इस्तेमाल होता है और कुछ भी store नहीं होता।

दो पहरेदार समय को ईमानदार रखते हैं:

- **एक device lock.** Process की हर timing, tuner की और cost probes की भी, एक ही lock रखती है: device
  साझा करने वाले दो मापन एक-दूसरे को माप लेते। एक ही key को tune करने वाले दो threads उसे एक बार
  मापते हैं; दूसरा पहले के जवाब का इंतज़ार करता है।
- **Stamp guard.** जो run अपने candidate के runs के median के आधे से कम हो वह खोया हुआ stamp है, और
  सबसे तेज़ रखने से पहले हटा दिया जाता है, tuner और probes दोनों के लिए।

## Store {#the-store}

| क्या | मान |
|---|---|
| Directory | `$SVOD_TK_TUNE_DIR`, वरना `$XDG_CACHE_HOME/svod/tk_tune`, वरना `~/.cache/svod/tk_tune` |
| File | हर device और crate version के लिए एक, जैसे `sm_86_28sm-v0.2.0.txt`, जिसकी शुरुआत उसके line format (`format 5`) से होती है; किसी दूसरे format वाली, या बिना इस line वाली फ़ाइल पूरी हटा दी जाती है |
| Line | `kind\|op\|device\|dtype\|shape\|candidates\|compiler\|programs numbers`: एक `choice` में चले हुए candidates होते हैं, सबसे तेज़ पहले, हर एक `index ns` (जिस candidate के programs वही emit करते हैं जो किसी और के करता है — एक unrolling जो schedule चुपके से मिटा देता है — वह अपने twin के ns पर दर्ज है, वह kernel जो दोनों बाँटते हैं); एक `value` में एक मापी गई संख्या; एक `admitted` में वे ranked indices जिन्हें admission ने रखा; एक `first` में पहले clean config का index |
| Key | `tune::TuneKey { op, device, dtype, shape, candidates, compiler }`, जहाँ `device` arch और SM count है (`sm_86-28sm`) और `compiler` device के compiler का digest है (`tune::compiler`) |

RTX 3060 पर एक असली store की lines: एक attention choice (shape `[batch, t, tk, heads, kv_heads, d]`,
candidate 0 3904 ns पर) और एक cost probe का value (हर unit पर femtoseconds)।

```text
choice|attention|sm_86-28sm|BFloat16|2x10x12x6x2x64|08bd45c243ff1581|9154cc72f03d25cd|ef103f71226add50 0 3904 1 3936
value|probe:cached_1m|sm_86-28sm|Int32||0a2d6c02b82988af|9154cc72f03d25cd|d2a8e296bcce85e4 16601
```

`programs` field उन कर्नेलों का fingerprint है जिन्हें candidates emit करते हैं: lowered program और
backend को सौंपा गया LLVM text, इसलिए कर्नेलों या renderer में बदलाव दोबारा मापता है और वही कर्नेल emit
करने वाला refactor नहीं। कोई program क्या emit करता है, यह उस चीज़ से याद रखा जाता है जिससे वह बना है,
executable के हर build के लिए, store के root के तहत `emitted/<build>.txt` में (एक हफ़्ते से इस्तेमाल न
हुआ build हटा दिया जाता है), इसलिए उसी build का warm process कुछ lower नहीं करता और rebuild हर program
को एक बार lower करता है। सिर्फ़ store miss compile करता है। Writes फ़ाइल को दोबारा पढ़ते हैं, merge करते
हैं और atomically बदलते हैं। न पढ़ा जा सकने वाला या न लिखा जा सकने वाला store miss गिना जाता है, कभी
error नहीं।

## Device costs {#device-costs}

Ranking की costs (`atoms::Costs`: एक shared byte, एक streamed byte, एक scattered element, पूरे SM के
तहत, अपने SIMD पर अकेले और warpgroup product के रूप में एक matrix-core step, एक score का softmax, दो ring
slots पर unrolled और rolled और एक slot पर एक pipeline trip, अपने SM पर अकेले block के दो और एक slot वाले
trips, एक lane move, एक shared round trip, दो ring slots से और जहाँ fill `cp.async` है वहाँ एक slot से
global latency, और L2 की दर और आकार) device पर `probe::costs` से तब मापी जाती हैं जब कोई process पहली बार उसके
लिए plan करता है। हर field programs की एक जोड़ी है, एक probe जो वह सब करता है जो उसका baseline करता है,
साथ में primitive की एक ज्ञात मात्रा। मान उसी store फ़ाइल में `value|probe:<field>` lines के रूप में जाते
हैं, ताकि बाद के processes उन्हें वापस पढ़ें। जो probe विफल हो, वह अपने field को शून्य पर छोड़ता है, उसकी
चेतावनी दी जाती है, और उसका नाम `Planner::unmeasured` में आता है, जिसे device gate
`device_costs_are_physical` ख़ाली रखता है: planner device को ग्राफ़ पर छोड़ने के बजाय जो उसके पास है उससे
rank करता है। सिर्फ़ बिना SM count वाले device के पास कोई planner नहीं होता।

Host tests इसके बजाय costs घोषित करते हैं (`test::unit::costs()`, और gfx1151 store से चिपकाई गई
`gfx1151_costs()`)। जिस कार्ड का device इस मशीन पर नहीं है, उसके लिए `Target::offline` या
`Target::card` target देता है, और `probe::stored` उस कार्ड से copy किए गए tune store से costs पढ़ता है;
उसके बिना `probe::GUESSED` उसकी जगह लेता है और ranking एक अनुमान है।
[`tk_dump` example](./testing-and-debugging#planning-for-another-card) दोनों करता है।

## Planner log {#the-planner-log}

`RUST_LOG=svod_tk::plan=debug` planner के फ़ैसले log करता है: ग्राफ़ लेने वाला op और क्यों, किसी
ranking के पहले 16 configs उनके अनुमानित ns के साथ और कितने configs उसने छोड़ दिए, store miss पर हर
config को compile करने से क्या दिखा (verdict, registers, spilled bytes), और tuned चुनाव।
`RUST_LOG=svod_tk=debug` tuner के rounds जोड़ता है।

## इसे बंद करना {#switching-it-off}

| कैसे | असर |
|---|---|
| `SVOD_TK_TUNE=0` | Candidate timing बंद; पहला ranked config जो clean compile हो वह चलता है (`first` line के रूप में रखा जाता है); cost probes फिर भी मापते हैं |
| `svod_tk::tune::scoped(false, f)` | इस thread पर (और जिन workers पर यह compile करता है उन पर) `f` के लिए वही, environment के ऊपर; tests इसे इस्तेमाल करते हैं |

`tune::TuneStore::at(root)` किसी दूसरे root पर store बनाता है, या `None` के साथ सिर्फ़ memory में।

## Probe tests {#probe-tests}

Probe tests `#[ignore]` tests हैं जो timings छापते हैं और कभी assert नहीं करते। वे device की probe की
गई costs से rank करते हैं और tuner की admit की हुई shortlist को मापते हैं, tuner के चुनाव को चिह्नित करते
हुए। इन्हें idle GPU पर एक-एक करके चलाएँ:

```bash
SVOD_DEVICE=CUDA cargo test -p svod-tk --release --lib gemm_candidates_probe -- --ignored --nocapture --test-threads=1
```

| Probe | क्या छापता है |
|---|---|
| `gemm_throughput_probe` | 4096³ bf16 पर admit हुए GEMM configs, TFLOP/s, और tuner का चुनाव |
| `gemm_candidates_probe` | Nemotron के projection shapes और 4096³ पर हर admit हुआ GEMM candidate: untuned चुनाव, सबसे तेज़ और tuner का चुनाव |
| `qwen3_gemm_candidates_probe` | Qwen3-Embedding के projections: shortlist, फिर lattice का बाक़ी हिस्सा |
| `attention_throughput_probe` | Flash attention (B 4, H 8, T 2048; d 64/128, causal और non-causal, bias के साथ d 64): हर admit हुआ config और tuner का चुनाव उसके softmax forms के साथ |
| `attention_candidates_probe` | Lattice का हर attention config, admit हुई shortlist के ख़िलाफ़ |
| `decode_throughput_probe` | Whisper large-v3 के decoder step का self और cross attention, tuned splits और बिना split |
| `conv_throughput_probe` | हर YOLO26 class के लिए हर admit हुआ convolution candidate, graph convolution के ख़िलाफ़ |
| `norm_bandwidth_probe` | 8192×1024 bf16 पर row kernels, GB/s में |
| `first_execution_probe` | tk GEMM को build, lower और prepare करने की host cost, graph GEMM के ख़िलाफ़ |
