---
sidebar_label: टेस्टिंग और डिबगिंग
---

# टेस्टिंग और डिबगिंग {#testing-and-debugging}

हर tk3 कर्नेल दो स्तरों पर जाँचा जाता है। Host पर, उसका tile program interpreter में एक सीधे reference
के ख़िलाफ़ चलता है, और lowered program हर target के लिए render होता है और, जहाँ toolchain installed हो,
compile होता है। tk3 kernels वाले GPU (CUDA या AMD) पर, lowered कर्नेल interpreter के ख़िलाफ़ या op के
अपने graph fallback के ख़िलाफ़ चलता है। Tests IR के hashes नहीं, बल्कि tolerances के भीतर values की तुलना
करते हैं।

## Interpreter {#the-interpreter}

`interp::run(&program, params, vars)` किसी tile program को host पर चलाता है। यह हर parameter के लिए एक
`Vec<f64>` लेता है और run के बाद हर parameter लौटाता है। हर block अपने statements क्रम से चलाता है, और
pipeline अपनी serial interleaving के रूप में चलती है। हर value अपने element type में round होती है
(`interp::round_to`), इसलिए नतीजा वही है जो एक सही lowering को accumulation order तक दोहराना चाहिए।
Scalar division शून्य की ओर truncate करता है, जैसा emitted code करता है। Runtime variables नाम से
bind होते हैं:

```rust
let out = run(&prog, vec![a.clone(), b.clone(), vec![0.0; m * n]], &[("b", 1)]).unwrap();
```

यह `interp::Error::{UnboundVar, ParamSize, Division, OutOfBounds}` लौटाता है: buffer की capacity से आगे
का access एक error है, चुपचाप होने वाला read नहीं। यह threads या barriers को model नहीं करता, इसलिए
ordering bugs सिर्फ़ device पर दिखते हैं।

## Test फ़ाइलें {#test-files}

| फ़ाइल (`test/unit/`) | स्तर | क्या जाँचता है |
|---|---|---|
| `atoms.rs`, `layout.rs`, `layouts.rs` | host | हर atom के operand layouts vendor ISA lane formulas के ख़िलाफ़, F2 algebra के नियम, inference, relayouts और swizzles |
| `build.rs`, `interp.rs`, `schedule.rs`, `sync.rs`, `emit.rs` | host | Builder records, interpreter semantics, pipeline expansion, barrier pass, emitted list |
| `ops_plan.rs` | host | हर op का `Plan` और error, हर shape, dtype और target के लिए |
| `offline.rs` | host | Copy किए गए tune store से पढ़े गए card targets और costs |
| `targets.rs` | host + device | हर target के लिए हर kernel family lower, lowering से पहले और बाद में interpret, और compile (`ptxas`, clang); GPU पर, device gate |
| `probe.rs` | host + device | हर cost probe हर target के लिए build होता है; GPU पर, `device_costs_are_physical` |
| `attention.rs`, `heads.rs`, `rows.rs`, `conv.rs`, `gather.rs` | host + device | Program एक f64 reference के ख़िलाफ़, फिर lowered कर्नेल program के ख़िलाफ़ |
| `device.rs`, `parts.rs` | device | GEMM शुरू से अंत तक, और वे हिस्से जिनसे attention बनता है (`cp.async` devices पर) |
| `ops.rs` | device | हर op अपने graph fallback के ख़िलाफ़, बेढब shapes पर |
| `tune.rs` | host + device | Store round trip, key की स्थिरता, एक असली GEMM tuning |

## Suite चलाना {#running-the-suite}

`SVOD_DEVICE` device चुनता है। Device tests `skipped: …` line के साथ सिर्फ़ वहीं skip होते हैं जहाँ चलाने
के लिए कुछ नहीं है: `SVOD_DEVICE=CPU`, ऐसा GPU जिसके लिए tk3 के पास kernels नहीं, या न `SVOD_DEVICE` और न
कोई supported GPU जुड़ा हुआ। Supported GPU जुड़ा हो और `SVOD_DEVICE` unset हो, तो वे fail होते हैं और वह
मान बताते हैं जो set करना है, क्योंकि जिस suite के सारे device tests skip हों वह pass होकर भी कुछ साबित
नहीं करता।

```bash
# Host only, no GPU:
SVOD_DEVICE=CPU cargo test -p svod-tk3 --release --lib
# On the GPU; on a fresh tune store run the cost probes alone first, since the parallel
# suite inflates what they measure and the store keeps the first measurement:
SVOD_DEVICE=CUDA cargo test -p svod-tk3 --release --lib device_costs_are_physical
SVOD_DEVICE=CUDA cargo test -p svod-tk3 --release --lib
```

`SVOD_DEVICE=AMD` (या `AMD:N`, KFD node क्रम में) एक AMD GPU चुनता है। कोई target तभी मापा हुआ गिना
जाता है जब उस पर `families_match_the_interpreter_on_the_device` pass हो जाए: device पर हर family का
पहला candidate interpreter के ख़िलाफ़। `register_staged_families_match_on_cuda` AMD data path
(register-staged fills, बिना `ldmatrix`) को CUDA GPU पर चलाता है।

### नए GPU पर {#on-a-new-gpu}

1. Tune store को किसी ऐसी directory पर रखें जिसे सँभालना है: `SVOD_TK3_TUNE_DIR=$PWD/tk3_tune_$(hostname)`।
2. कई AMD nodes वाली मशीन पर जाँचें कि `SVOD_DEVICE=AMD:N` जिस node का नाम लेता है वह GPU ही है
   (`/sys/class/kfd/kfd/topology/nodes/*/properties`, `gfx_target_version`)।
3. पहले `device_costs_are_physical` अकेले चलाएँ, फिर `families_match_the_interpreter_on_the_device`,
   फिर suite। जिस device का gate fail हो उसके आँकड़े रिपोर्ट न करें।

Nemotron के लिए model-level gate (असली weights, generated goldens):

```bash
SVOD_DEVICE=CUDA cargo test -p svod-model --release --lib nemotron_diar::parity::half_precision -- --ignored --nocapture --test-threads=1
```

## क्या कर्नेल चला? {#did-the-kernel-run}

जो op fallback करता है वह `CALL` की जगह graph ops बनाता है; `ops.rs` tests नतीजे के ग्राफ़ की calls में
कर्नेल का नाम ढूँढते हैं। बिना device के path का अनुमान लगाने के लिए `ops::shape` planner को बुलाएँ
(देखें [Op Layer](./op-layer#kernel-or-graph)); device पर, planner log बताता है कि op ने ग्राफ़ क्यों
लिया ([Tuning](./tuning#the-planner-log))।

## किसी दूसरे कार्ड के लिए planning {#planning-for-another-card}

`tk3_dump` example दिखाता है कि tk3 ऐसे कार्ड पर क्या चलाएगा जिसका device इस मशीन पर नहीं है: अनुमानित
ns के साथ planner की ranking और lattice का हर config क्यों छोड़ा गया (`plan.txt`), चुने गए config का LLVM
text और, जहाँ tools installed हों, उसका PTX, `ptxas -v` resources और SASS (CUDA) या अपनी resource
comments के साथ AMDGCN assembly (AMD), और main loop का एक opcode histogram।

```bash
cargo run -p svod-tk3 --release --example tk3_dump -- --card h100-pcie --op gemm --shape 4096x4096x4096
cargo run -p svod-tk3 --release --example tk3_dump -- --arch gfx942 --op attention --shape 8x16x1024x128 --causal --store DIR
```

`--card` `atoms::CARDS` का एक नाम लेता है (`rtx3060`, `h100-pcie`, `h100-sxm`, `rtx5070`,
`rtx5090`, `mi300x`, `mi355x`, `rx9070xt`, `strix-halo`); `--arch` `--sms`/`--xcds` के साथ या सूची
के पहले कार्ड वाले मानों के साथ एक architecture लेता है। `--store` कार्ड से copy की गई tune store
directory है, जिसकी probe lines ranking की कीमत तय करती हैं; उसके बिना guessed costs यह करती हैं।
`--cfg N` N-वाँ ranked config emit करता है, `--out DIR` output directory का नाम देता है (`tk3_dump`)।

## Environment variables {#environment-variables}

| Variable | असर |
|---|---|
| `SVOD_DEVICE` | Default device: `CUDA[:N]`, `AMD[:N]`, `CPU`। यह तय करता है कि kernels लागू होते हैं या नहीं और कौन से device tests चलते हैं |
| `SVOD_TK3_TUNE=0` | कोई candidate timing नहीं; पहला clean config चलता है (देखें [Tuning](./tuning)) |
| `SVOD_TK3_TUNE_DIR` | Tune store की directory |
| `RUST_LOG=svod_tk3::plan=debug` | Planner के फ़ैसले; `svod_tk3=debug` tuner के rounds जोड़ता है |
| `TK3_DUMP_LIST=1` | Program lower होने पर हर emitted instruction छापता है (`[i] id op dtype <- sources`)। Bodies memoize होती हैं, इसलिए यह हर program की सिर्फ़ पहली lowering छापता है |
| `SVOD_SPEC_DEBUG=1` | जब कोई program IR spec verification में fail हो, तो ठुकराया गया instruction और उसका tree छापता है |

## किसी मॉडल की profiling {#profiling-a-model}

Nemotron example एक बार warm up करता है (जिससे tune store भी भर जाता है), फिर दूसरे run को मापता और
profile करता है, और उसकी per-kernel report छापता है:

```bash
cargo run -p svod-model --release --example nemotron_diarize -- audio_1.wav --dtype bf16 --profile
```

tk3 kernels अपने कर्नेल नामों (`gemm`, `conv`, `flash_attention`, `heads`, `layer_norm`, …) के तहत
graph kernels के बगल में दिखते हैं, उसी तरह मापे हुए।

:::tip[शांत GPU पर मापें]
GPU पर कोई दूसरा process हर आँकड़े को बिगाड़ देता है। Probes और tune store पहले clock को warm करते हैं।
Probes को `--test-threads=1` के साथ चलाएँ।
:::
