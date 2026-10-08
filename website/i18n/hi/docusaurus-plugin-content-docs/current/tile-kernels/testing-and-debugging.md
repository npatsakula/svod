---
sidebar_label: टेस्टिंग और डिबगिंग
---

# टेस्टिंग और डिबगिंग

हर tk3 कर्नेल दो स्तरों पर जाँचा जाता है। Host पर उसका tile program interpreter में एक सीधे reference
के ख़िलाफ़ चलता है। CUDA device पर lowered कर्नेल interpreter के ख़िलाफ़ या op के अपने graph fallback के
ख़िलाफ़ चलता है। Tests tolerances के भीतर values की तुलना करते हैं, IR के hashes की नहीं।

## Interpreter {#the-interpreter}

`interp::run(&program, params, vars)` एक tile program को host पर चलाता है। यह हर parameter के लिए एक
`Vec<f64>` लेता है और run के बाद हर parameter लौटाता है। हर block अपने statements क्रम से चलाता है, और
pipeline अपने serial interleaving के रूप में चलती है। हर value अपने element type पर round होती है
(`interp::round_to`), इसलिए नतीजा वही है जो एक सही lowering को accumulation order तक दोहराना चाहिए।
Runtime variables नाम से bind होते हैं:

```rust
let out = run(&prog, vec![a.clone(), b.clone(), vec![0.0; m * n]], &[("b", 1)]).unwrap();
```

यह `interp::Error::{UnboundVar, ParamSize, Raw}` लौटाता है। यह threads या barriers को model नहीं करता,
इसलिए ordering bugs सिर्फ़ device पर दिखते हैं।

## Test फ़ाइलें {#test-files}

| फ़ाइल | स्तर | क्या जाँचती है |
|---|---|---|
| `layout.rs`, `layouts.rs`, `atoms.rs` | host | F2 algebra के नियम, हर atom उसके closed form के ख़िलाफ़, inference के नतीजे |
| `interp.rs`, `schedule.rs`, `build.rs` | host | Interpreter semantics, pipeline expansion |
| `ops_plan.rs` | host | हर shape, dtype और target पर हर op का `Plan` और error |
| `attention.rs`, `heads.rs`, `rows.rs` | host + device | f64 reference के ख़िलाफ़ program, फिर program के ख़िलाफ़ lowered कर्नेल |
| `device.rs`, `parts.rs` | device | GEMM (हर config जो op layer चुन सकती है) और attention के हिस्से, interpreter के ख़िलाफ़ |
| `ops.rs` | device | अटपटे shapes पर हर op उसके graph fallback के ख़िलाफ़, tuning बंद करके |
| `tune.rs` | host + device | Store round trip, key stability, एक असली GEMM tuning |

Device tests एक `skipped: no CUDA device` line के साथ skip होते हैं जब तक `SVOD_DEVICE` किसी CUDA device
का नाम न ले। पूरा suite GPU पर serialized चलाएँ:

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk3 --lib --release -- --test-threads=1
```

Nemotron के लिए model-level gate (असली weights, generated goldens):

```bash
SVOD_DEVICE=CUDA:0 cargo test -p svod-model --release --lib nemotron_diar::parity::half_precision -- --ignored --nocapture --test-threads=1
```

## क्या कर्नेल चला? {#did-the-kernel-run}

जो op fallback करता है वह `CALL` की जगह graph ops बनाता है। `ops.rs` tests नतीजे के calls में कर्नेल
का नाम खोजते हैं:

```rust
fn assert_kernel(t: &Tensor, name: &str) {
    let calls: Vec<String> = t
        .uop()
        .toposort()
        .iter()
        .filter_map(|u| match u.op() {
            Op::Call(ops::Call { info, .. }) => info.name.clone(),
            _ => None,
        })
        .collect();
    assert!(calls.iter().any(|n| n == name), "no {name} kernel among {calls:?}");
}
```

बिना device के path का अनुमान लगाने के लिए `ops::shape` planner बुलाएँ (देखें [Op Layer](./op-layer#kernel-or-graph))।

## Environment variables {#environment-variables}

| Variable | असर |
|---|---|
| `TK3_DUMP_LIST=1` | Program lower होने पर हर emit किया गया instruction छापता है (`[i] id op dtype <- sources`)। Bodies memoize होती हैं, इसलिए यह हर program की सिर्फ़ पहली lowering छापता है |
| `SVOD_SPEC_DEBUG=1` | जब कोई program IR spec verification में फ़ेल हो, तो ठुकराया गया instruction और उसका tree छापता है |
| `SVOD_TK3_TUNE=0` | कोई मापन नहीं; पहला candidate चलता है (देखें [Tuning](./tuning)) |
| `SVOD_DEVICE=CUDA:0` | GPU पर चलाएँ (default device तय करता है कि कर्नेल लागू होंगे या नहीं) |

## किसी मॉडल की profiling {#profiling-a-model}

Nemotron example एक बार warm up करता है (जिससे tune store भी भर जाता है), एक run का समय मापता है, फिर
तीसरे run की per-kernel report छापता है:

```bash
cargo run -p svod-model --release --example nemotron_diarize -- audio_1.wav --dtype bf16 --profile
```

tk3 कर्नेल अपने नामों (`gemm`, `flash_attention`, `heads`, `layer_norm`, …) के तहत graph kernels के साथ
दिखते हैं, उसी तरह मापे जाते हुए।

:::tip[शांत GPU पर मापें]
RTX 3060 idle में 210 MHz पर रहता है, और GPU पर कोई दूसरा process हर संख्या बिगाड़ देता है। Probes
और tune store पहले clock को warm करते हैं। Probes `--test-threads=1` के साथ चलाएँ।
:::
