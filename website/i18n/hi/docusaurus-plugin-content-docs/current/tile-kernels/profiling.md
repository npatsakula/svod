---
sidebar_label: Profiling
---

# Thread trace के साथ profiling {#profiling-with-a-thread-trace}

किसी कर्नेल का समय आपको बताता है कि वह धीमा है। Thread trace बताती है कि क्यों: हर wave किस
instruction पर रुकी, कितनी देर, और वह instruction tile program की किस line से आया। यह पेज आपको
"यह कर्नेल धीमा है" से tk program में एक बदलाव तक ले जाता है।

## Trace क्यों {#why-a-trace}

बाक़ी tools कर्नेल को बाहर से देखते हैं। Device time बताता है कि उसमें कितना समय लगा, roofline बताता है
कि वह memory और compute की सीमाओं से कितना दूर है, और counters बताते हैं कि units कितनी व्यस्त थीं।
इनमें से कोई नहीं बताता कि किसी दिए गए cycle पर कोई wave क्या कर रही थी। AMD के RDNA cards पर SQ
thread trace (SQTT) ठीक यही record करती है, इसलिए आप देख सकते हैं कि main loop global loads का, LDS
का, किसी barrier का इंतज़ार करता है, या किसी का भी नहीं।

Svod के profiler के पाँच tiers हैं। हर एक अपने नीचे वाले में detail जोड़ता है:

| Tier | क्या देता है | कहाँ |
|---|---|---|
| 1, device time | GPU clock पर हर कर्नेल का समय | GPU backends; CPU पर wall clock |
| 2, roofline | Tier 1 और कर्नेल के static work estimate से GFLOP/s और GB/s | हर backend |
| 3, static occupancy | Registers, LDS, scratch और एक occupancy estimate, बिना dispatch के | AMD, CUDA; Metal में आंशिक रूप से |
| 4, hardware counters | Busy cycles, waves, instructions (`SVOD_PMC`) | AMD (PM4), CUDA (CUPTI) |
| 5, thread trace | हर traced wave के instructions उनके cycles के साथ, एक अतिरिक्त pass में | AMD gfx11 और gfx12 |

## Trace क्या record करती है {#what-the-trace-records}

Trace हर shader engine (SE) के लिए tokens की एक stream है: कोई wave कब शुरू हुई और कब ख़त्म हुई, और
traced SIMD के लिए, उसने कौन-सा instruction कब issue किया। Svod stream को decode करता है, हर
instruction को कर्नेल के disassembly से वापस map करता है, और main loop के cycles को buckets में बाँटता
है:

- **issue**: instruction issue हो रहा था, अपनी class के हिसाब से (`matrix`, `valu`, `salu`, `lds`, `vmem_load`, …);
- **stall**: वह तैयार था लेकिन उसकी pipe या queue व्यस्त थी;
- **wait**: एक `s_wait_*` ने किसी load, LDS access या store का इंतज़ार किया, जो उस load के नाम लिखा जाता है जिसने उसे बाँधा;
- **barrier**: wave किसी workgroup barrier पर रुकी हुई थी;
- **idle**: कोई record नहीं, क्योंकि SIMD किसी दूसरी wave के लिए issue कर रहा था या किसी ऐसी dependency पर था जिसे trace नहीं दिखाती।

हर bucket का एक **exposed** हिस्सा भी होता है: उसके cycles, उसी समय दूसरी pipes के काम ने जितना छिपाया
उसे घटाकर। 180 cycles का load wait, जिनमें से 140 के दौरान matrix pipe व्यस्त है, लगभग 40 का पड़ता है।
Findings exposed हिस्से से rank होती हैं, क्योंकि सिर्फ़ exposed cycles हटाने पर ही कर्नेल छोटा होता है।

एक **trip** tile program के loop body का एक pass है, जो loop के unrolled होने पर hardware iteration से
छोटा हो सकता है। Cycles किसी traced wave के होते हैं, प्रति trip।

## Scope और cost {#scope-and-cost}

- **Hardware.** AMD gfx11 (RDNA3, RDNA3.5) और gfx12 (RDNA4)। कोई भी दूसरा target
  `only gfx11 and gfx12 are traced` के साथ मना कर दिया जाता है। Captures gfx1201 (RX 9070 XT) और gfx1151
  (Strix Halo) पर verify की गईं। gfx1100 की traces decode होती हैं, लेकिन कोई hardware capture verify
  नहीं की गई।
- **gfx11 पर कोई ceiling figure नहीं।** gfx11 matrix op को एक-cycle token के रूप में record करता है,
  इसलिए वहाँ pipes के busy हिस्से अज्ञात हैं। Report कोई `ceiling` line नहीं छापती, और उसकी findings
  कच्चे (raw) हिस्से इस्तेमाल करती हैं, जिन पर `approx: untimed_matrix` का निशान होता है।
- **सिर्फ़ PM4 queues.** `SVOD_AMD_AQL=1` के साथ खोला गया device trace को arm करने से मना कर देता है।
- **Window.** हर SE को default में 64 MiB का buffer मिलता है (`SqttConfig::window`, 4 KiB pages में,
  configurable)। जो buffer भर जाता है वह अपना tail खो देता है। तब report `!! TRUNCATED` से शुरू होती है,
  और उसके totals कम गिनते हैं।
- **हर traced कर्नेल के लिए एक अतिरिक्त pass.** Profile अपने runs का समय बिना trace के मापता है, फिर
  trace arm करके plan को एक बार और चलाता है। वह pass हर कर्नेल के बाद synchronize करता है और अगले से
  पहले उसके buffers free करता है। कोई timing column उससे नहीं आता।
- **दो SIMDs.** Instruction tokens SE0 और SE1 से आते हैं, हर एक का एक SIMD (`SqttConfig::se_mask`)।
  `SVOD_SQTT_SIMD` उन्हें चुनता है: हर SE के लिए एक SIMD, या एक comma list जो SEs पर दोहराई जाती है
  (default `0,1`, SE0 पर SIMD 0 और SE1 पर SIMD 1)। Per-instruction संख्याएँ उन्हीं दो SIMDs की waves के
  बारे में हैं।

## Clocks और prerequisites {#clocks-and-prerequisites}

- ऊपर के targets में से एक AMD card, `SVOD_DEVICE=AMD` (या `AMD:N`) से चुना हुआ, PM4 queues पर चलता हुआ।
- उस clang के बगल में `llvm-objdump` जिससे Svod compile करता है। Capture अपना disassembly सहेजती है,
  इसलिए replay को tool की ज़रूरत तभी होती है जब `kernel.s` मौजूद न हो।
- **Unpinned clock, default.** Traced pass उसी clock पर चलता है जो device चुनता है, ठीक वैसे ही जैसे
  untraced run। `SVOD_SQTT_PSTATE=standard` (या `peak`) DRM context के ज़रिए एक stable power state pin
  करता है। यह shader clock घटाता है और trace जो दिखाती है उसे बिगाड़ता है। इसे सिर्फ़ किसी पुरानी pinned
  capture से मिलान के लिए इस्तेमाल करें।
- एक शांत GPU. Tracing हर हाल में run को बदलता है: report traced run का समय untraced के अनुपात में छापती
  है, और 1.2× से ऊपर वह findings पर `approx: perturbed` का निशान लगाती है। Traced समय की तुलना कभी
  untraced समय से न करें।

## Rust से tracing {#tracing-from-rust}

`ProfileOptions::sqtt` किसी भी profiled plan (`ExecutionPlan::profile`) या tensor (`Tensor::profile`)
के लिए tier 5 चालू करता है:

```rust
use svod_runtime::{ProfileOptions, SqttOptions};

let opts = ProfileOptions { sqtt: Some(SqttOptions::default()), ..Default::default() };
let profile = plan.profile(&opts)?;
for traced in profile.stages.iter().flat_map(|stage| &stage.sqtt) {
    match &traced.analysis {
        Ok(analysis) => println!("{}: {}", traced.kernel, analysis.summary()),
        Err(error) => println!("{}: {error}", traced.kernel),
    }
}
```

`SqttOptions` में एक `SqttConfig` (SEs, SIMDs, window, power state) और एक `Sink` होता है:

- `Sink::Disk(dir)`, default, हर कर्नेल की capture `dir/<index>_<entry point>` के तहत सहेजता है
  (default में `./sqtt`), इसलिए pass एक समय में एक capture रखता है। `Traced::raw` `Raw::Saved(path)` है।
- `Sink::Memory` हर capture `Raw::Held` में रखता है। एक capture प्रति SE 64 MiB तक ले सकती है, इसलिए
  पूरे मॉडल की captures शायद न समाएँ।

हर `Traced` अपना `analysis` ख़ुद रखता है। जिस कर्नेल की trace analyze नहीं होती, वह बाक़ियों को अछूता
छोड़ता है। `RunProfile::render_report` हर traced कर्नेल के लिए एक summary line के साथ ख़त्म होता है।

## किसी मॉडल की tracing {#tracing-a-model}

Model examples का `--profile` flag trace नहीं करता। किसी मॉडल को trace करने के लिए, उसके output tensor
को ऊपर के options के साथ profile करें। Plan का हर compiled कर्नेल trace होता है: tk kernels और graph
kernels दोनों। `SqttOptions::from_env()` उस program के लिए जिसे switch चाहिए, options environment से
बनाता है: `SVOD_SQTT=1` trace करता है, `SVOD_SQTT_DIR` directory का नाम देता है, और `SVOD_SQTT_PSTATE`
और `SVOD_SQTT_SIMD` config तय करते हैं। जब `SVOD_SQTT` `1` नहीं होता तो यह `Ok(None)` लौटाता है, और जो
मान parse नहीं होता उसके लिए error।

Untraced run की per-kernel table से शुरू करें, जो कर्नेल हावी है उसे चुनें, और उसकी directory
`tk_bench sqtt --load` (नीचे) से पढ़ें। tk कर्नेल के लिए, अगला क़दम उसे अकेले trace करना है, ताकि आप
उसका config बदल सकें।

## एक कर्नेल की tracing {#tracing-one-kernel}

`tk_bench sqtt` एक shape पर एक tk op बनाता है, हर config का समय tuner की तरह बिना trace के मापता है,
उसे एक बार trace करता है, capture सहेजता है और उसकी report छापता है:

```bash
SVOD_DEVICE=AMD cargo run --release -p svod-tk --example tk_bench -- sqtt nearest 4096 128 16
```

| Command | क्या trace करता है |
|---|---|
| `sqtt gemm M N K [--residual] [--gated]` | एक bf16 GEMM, residual या SwiGLU epilogue के साथ |
| `sqtt attention B H T D [--causal] [--bias] [--flip FORM,…] [--ring S]` | `[B, T, H, D]` पर bf16 attention; `--flip` हर config के softmax forms को पलटता है, `--ring` उसके ring slots तय करता है |
| `sqtt nearest N D K [--metric M] [--splits S]` | depth `D` पर `N` f16 rows का self-join, हर row के लिए `K` रखते हुए; `--splits` config के splits को override करता है |
| `sqtt conv H W CIN COUT KERNEL STRIDE [--batch B] [--residual]` | bias और SiLU के साथ किसी YOLO block का f16 convolution |

चारों `--cfg`, `--top N` (table rows, default में 12), `--out DIR` (default में `./sqtt`) और `--json`
(report एक JSON line के रूप में) लेते हैं। `--cfg` एक comma list है। `tuned`, default, वह config है जो
op launch करेगा। एक संख्या `I` planner का `I`वाँ ranked config है। GEMM एक pinned code
`BMxBNxBK:S:WRxWC[:rolled]` भी लेता है, और `nearest` और `conv` `ranked` लेते हैं, जो trace करने के
बजाय ranking की सूची देता है। हर config `DIR/<op>_<shape>_<cfg>/p<program>/<index>_<entry>/` में
पहुँचता है, और path `saved …` के रूप में छपता है।

## सहेजी गई directory और replay {#the-saved-directory-and-replay}

| File | क्या रखती है |
|---|---|
| `se0.att`, `se1.att`, … | हर SE की raw trace stream |
| `kernel.co` | Traced code object; AMD पर launch हुआ tk कर्नेल अपने DWARF में अपनी `kernel.tk` listing पर एक line table, और ख़ुद listing भी, रखता है |
| `kernel.s` | `kernel.co` का `llvm-objdump -d`; report के PCs इसके addresses हैं |
| `meta.json` | `format` 2, arch, load base, untraced और traced ns, SEs, labels (op, shape, cfg), loop model, resources, workgroup और grid, power state, memory clock |

Load `format` जाँचता है और अलग होने पर आपसे फिर से capture करने को कहता है। Statements हर load पर
`kernel.co` से वापस पढ़े जाते हैं। सिर्फ़ AMD tk kernels के पास ये होते हैं; graph kernels के पास कोई
नहीं (CUDA और Metal के पास कोई thread trace नहीं)। Statements के बिना, एक trip को एक loop iteration
माना जाता है, जिस पर `approx: per_iteration` का निशान होता है।

Replay को किसी GPU की ज़रूरत नहीं:

```bash
cargo run --release -p svod-tk --example tk_bench -- sqtt --load DIR [--json] [--attribute]
```

Rust से, `svod_runtime::sqtt::Analysis::load(dir)` वही analysis लौटाता है।

## Report पढ़ना {#reading-the-report}

`rdna4_nearest_split` fixture (`device/src/test/fixtures/sqtt/svod/`) एक gfx1201 nearest कर्नेल है,
n4096 d128 k16। संक्षेप में, उसकी report ऐसी है:

```text
== verdict: cycles are a traced wave's, a trip (1 an iteration, its statements' unroll copies); exposed: not hidden by other pipes' work
ceiling    matrix pipe 14.7% busy over the steady window: removing every stall gains at most 6.82x; …
bound      occupancy_limited  56.6%  SIMD idle 56.6%: 2.0 waves a SIMD, ~4.6 would fill it; limit 8 by vgpr+lds
   knob: fewer waves resident than the kernel allows: more workgroups in flight
 1 barrier            18.6% (raw 23.4%)  barrier (sink): 255/trip parked for the workgroup's slowest wave, 203 exposed; …
   knob: drop or merge a barrier only if its exposed share passes 10%
 2 load_exposed       13.0% (raw 16.8%)  cycles a trip: waiting on loads 183 (141 exposed); stalled issuing 0 (0 exposed)
   0x364 load exposed 129/trip at wait 0x44c (then ds_store_b128 v177, v[105:108]): cover med 91, latency med 238, … @21 fill c staged issue [steady]
   knob: latency: more work between a load and its wait (prefetch distance, ring slots)
 3 loop_overhead       2.7% (raw 3.2%)  valu 26 + salu 9 issue cycles a trip beside matrix 146 (24%), …
kernel     144840 ns untraced, 148360 ns traced (1.02x), wave span 461319 cycles, sclk ~3109 MHz, mclk 875 MHz, pstate unpinned
traced     se0 sa0 wgp0 simd0 (2 waves), se1 sa0 wgp0 simd1 (2 waves) of 2 SEs
== statements: main loop cycles a trip by kernel.tk line (a wait on its binding load's); 104 of 105 loop instructions attributed; top 12 by exposed
 line  pcs    issue    stall     wait  barrier     idle    total  exposed  statement
 1581    1      0.0      0.0      0.0    255.1      0.0    255.1    202.6  await [steady]
   21    8      7.0      2.1    182.7      0.0     32.7    224.5    162.5  fill c staged issue [steady]
   28   24    145.7     11.4      0.0      0.0     75.5    232.6    148.0  mma x·c_s [steady]
```

इसे ऊपर से नीचे पढ़ें:

1. **`ceiling`**: matrix pipe कितनी व्यस्त थी। 14.7% पर यह loop matrix की सीमा से बहुत दूर है, और हर
   stall हटाने से अधिकतम 6.82× का लाभ होगा।
2. **`bound`**: कर्नेल की class, pipes के busy हिस्सों से (`matrix_bound`, `issue_bound` या
   `occupancy_limited`)।
3. **क्रमांकित findings**: loop के exposed cycles किसमें गए, सबसे बड़ा पहले। हर एक अपना evidence
   (PCs, किसी wait के पीछे का load, barrier पर आख़िर में कौन पहुँचा), `@` के बाद tk statement, और क्या
   बदलना है यह बताने वाली एक `knob` line देती है।
4. **`kernel` और `traced`**: untraced और traced समय, clock, और कौन-से SIMDs trace हुए।
5. **`statements`**: `kernel.tk` line के हिसाब से loop के प्रति trip cycles। `[steady]` loop की steady
   state है; `[prologue]`, `[head]`, `[drain]` और `[epilogue]` उसके आसपास के हिस्से हैं।

इसके बाद और sections आते हैं: class के हिसाब से समय, प्रति SE occupancy, bucket के हिसाब से main loop के
प्रति trip cycles, शीर्ष stalls, waits उन loads के साथ जिन्होंने उन्हें बाँधा, और सबसे गर्म instructions।

## Verdict classes और उनके levers {#verdict-classes-and-their-levers}

Thresholds heuristics हैं, जो `runtime/src/sqtt/verdict.rs` में constants के रूप में रखे गए हैं।

| Class | कब fire होती है | Knob | tk lever |
|---|---|---|---|
| `matrix_bound` | Matrix pipe steady window के ≥ 70% व्यस्त | `less_matrix_work` | सस्ता type या कम matrix ops; schedule समस्या नहीं है |
| `issue_bound` | कोई pipe ≥ 90% व्यस्त, उसका matrix 60% से कम | `cut_issue` | Loop से non-matrix instructions हटाएँ |
| `occupancy_limited` | SIMDs ≥ 10% idle | `free_resources`, `more_workgroups`, `more_waves` | प्रति workgroup कम VGPRs या कम LDS; ज़्यादा workgroups के लिए छोटा tile या ज़्यादा splits |
| `load_exposed` | Global load waits और stalls loop के ≥ 10%, exposed | `latency`, `move_use` | गहरी ring या एक step पहले issue किया गया load; `move_use` load के ठीक पीछे के use का नाम देता है |
| `lds_exposed` | LDS waits और stalls ≥ 10%, exposed | `lds_flush`, `lds_use`, `lds_queue` | कम और चौड़े `ds` ops, अगला fragment पहले पढ़ा गया, conflict-free layout |
| `barrier` | Barrier cycles loop के ≥ 10% | `barrier` | पहले दूसरी findings ठीक करें; barrier को हटाएँ, merge या split करें सिर्फ़ तब जब उसका exposed हिस्सा 10% पार करे |
| `loop_overhead` | VALU+SALU issue matrix issue के ≥ 10%, या कोई भी divergent branch | `hoist` | Address और guard arithmetic को hoist करें; guards को uniform बनाएँ |

Barrier एक sink है: waves सबसे धीमी wave के लिए वहाँ रुकती हैं, इसलिए उसका समय उसी चीज़ का है जिसने
उस wave को धीमा किया। Finding बताती है कि आख़िर में कौन पहुँचा और किस instruction के बाद।

`approx` के निशान वाली finding के आँकड़े सावधानी से पढ़ने हैं: `perturbed`, `untimed_matrix`,
`approx_bindings` (अनुमानित binding), `implausible_time` (corrupt stream), `truncated` या
`per_iteration`।

## `--attribute` और `--compare` {#--attribute-and---compare}

`--attribute` सिर्फ़ GEMM के लिए है, क्योंकि सिर्फ़ `tk_bench sqtt gemm` capture में cost model की price
होती है। यह हर priced term (`steps`, `reads`, `fill`, `wait`, `lone_trip`, `trip`) को उन instructions
के traced cycles के बगल में रखता है जिनकी price वह लगाता है। Terms overlap करते हैं, इसलिए totals की
नहीं, हिस्सों और प्रति trip गति की तुलना करें।

`--compare A B` B की capture को A की capture के सामने रखता है (Δ = B − A):

```bash
cargo run --release -p svod-tk --example tk_bench -- sqtt --compare A B [--json] [--top N]
```

यह दोनों समय छापता है, वे findings जो प्रकट हुईं, ग़ायब हुईं या बदलीं, |Δ exposed| के हिसाब से loop के
प्रति trip buckets, text के हिसाब से statements, और mnemonic के हिसाब से aligned loop के instructions।
यह `!!` के साथ चेतावनी देता है जब दोनों captures अलग power states पर चलीं, जब उनके traced/untraced
अनुपात 10% से ज़्यादा अलग हैं, या जब उनके trips reduction की अलग-अलग मात्राएँ cover करते हैं। जिस
capture में कोई main loop नहीं, उसके पास align करने को कुछ नहीं।

## Workflow {#the-workflow}

1. Untraced profile की per-kernel table में धीमा कर्नेल ढूँढें।
2. उसे trace करें: `tk_bench sqtt <op> <shape>`, या ऐसे कर्नेल के लिए जो सिर्फ़ किसी मॉडल के भीतर मौजूद
   है, `SqttOptions` के साथ `Tensor::profile`।
3. `ceiling` और `bound` पढ़ें। `matrix_bound` कर्नेल अपनी सीमा के पास है: schedule नहीं, काम बदलें।
4. पहली finding, उसका evidence और उसका `@` statement लें। वह statement tile program की वह line है जिसे
   बदलना है।
5. एक बदलाव करें, `--out` को एक नई directory पर set करके फिर से trace करें, और `--compare old new`
   चलाएँ। जाँचें कि लक्षित bucket हिला, और लाभ को सिर्फ़ untraced समय से आँकें।
6. बदलाव रखने से पहले idle GPU पर `tk_bench` या tuner से पुष्टि करें।

## हल किए गए उदाहरण {#worked-examples}

### Attention: bias को एक step पहले load करना {#attention-loading-the-bias-a-step-ahead}

सीधे पढ़ी जाने वाली attention bias tile उसी step में load होती थी जिसके scores उसे इस्तेमाल करते थे,
इसलिए उसका wait load के ठीक पीछे बैठता था। यही वह pattern है जिसे `load_exposed` `move_use` के साथ नाम
देता है। अब कर्नेल हर tile को एक step पहले load करता है (`tk/src/kernels/attention.rs` में
`pipeline_with` उसे अगले step में ले जाता है), इसलिए load के पास पहुँचने के लिए पूरा एक step होता है।
PR #230 ने d128 पर −19% और d64 पर −7% मापा।

### Nearest: RDNA4 पर barrier को split करना {#nearest-splitting-the-barrier-on-rdna4}

`rdna4_nearest_joined` और `rdna4_nearest_split` fixtures एक ही `NearestCfg` (n4096 d128 k16) हैं,
RDNA4 के split barrier से पहले और बाद। gfx12 पर register-staged ring का steady loop अब LDS commit के
बाद barrier को signal करता है (`arrive`, `s_barrier_signal`) और trip के अंत में उसका इंतज़ार करता है
(`await`, `s_barrier_wait`)। दोनों के बीच का tail तब चलता है जब धीमी waves पीछे से पहुँच रही होती
हैं। Fixtures पर `--compare` यह दिखाता है:

```text
time       180320 -> 144840 ns untraced (-19.7%); traced/untraced 1.01x -> 1.02x; …
changed    barrier            46.8% ->  18.6%
appears    load_exposed           - ->  13.0%
bucket                A raw    B raw    Δ raw    A exp    B exp    Δ exp
barrier              1081.7    255.1   -826.6    962.9    202.6   -760.2
   A raw    B raw    Δ raw    A exp    B exp    Δ exp  statement
  1081.7      0.0  -1081.7    962.9      0.0   -962.9  barrier [steady]
     0.0    255.1   +255.1      0.0    202.6   +202.6  await [steady]
```

Barrier exposure प्रति trip 963 से 203 cycles पर गिरता है, `barrier [steady]` statement `arrive` +
`await [steady]` बन जाता है, और untraced समय 180 से 145 µs पर जाता है। हर पक्ष एक अकेला traced run है।
PR #230 के benchmark ने nearest d128 k16 के लिए −5.6% मापा। जो `load_exposed` finding प्रकट होती है वह
अगला lever है: barrier के जाने के बाद global loads की latency दिखती है।

### GEMM: एक ragged M {#gemm-a-ragged-m}

`rdna4_gemm_1536` और `rdna4_gemm_1500` fixtures M 1536 और M 1500 पर एक ही GEMM config हैं (N 1280,
K 5120, Whisper का fc2)। दोनों replays `bound matrix_bound` कहते हैं। सिर्फ़ 1500 वाला `loop_overhead`
जोड़ता है, जिसका evidence main loop में आठ divergent branches हैं:

```text
 2 loop_overhead       0.5% (raw 0.7%)  valu 24 + salu 7 issue cycles an iteration beside matrix 443 (7%), 22 valu ops an iteration, 8 divergent branches (approx: per_iteration)
   0x5f0 s_and_saveexec_b32 s1, vcc_lo
   0x5f4 s_cbranch_execz 3
   knob: hoist address and guard arithmetic out of the loop; make guards uniform (no exec branches)
```

जो M tile का multiple नहीं है वह edge guards को loop के भीतर exec branches के रूप में डाल देता है, और
1536 वाली capture, जो अपनी 128-row tile का multiple है, में कोई नहीं है। सबक़ गुणात्मक है: ragged edge
loop instructions की कीमत लेता है जो tile तक padded shape नहीं चुकाता, इसलिए ragged shape के schedule
को tune करने से पहले padded shape जाँचें।

## Caveats {#caveats}

- Trace दो SIMDs देखती है। जो grid उन पर कभी नहीं पहुँचता वह वहाँ कुछ trace नहीं करता।
- Traced समय कभी benchmark नहीं है। किसी बदलाव को `tk_bench` या tuner से आँकें, बिना trace के।
- `exposed` इसका अनुमान है कि दूसरी pipes ने क्या छिपाया, इसलिए यह उस समय का वादा नहीं है जो कोई fix
  बचाता है।
- Thresholds heuristics हैं। 10% से कम की finding evidence है, फ़ैसला नहीं।
- Statements के बिना (graph कर्नेल, या ऐसी capture जिसके code object में कोई line table नहीं), आँकड़े
  प्रति trip के बजाय प्रति loop iteration होते हैं।
- पुराने `meta.json` format की capture load नहीं होती; उसे फिर से capture करें।
