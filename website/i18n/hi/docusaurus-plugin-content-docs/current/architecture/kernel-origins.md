---
sidebar_label: कर्नेल Origins
---

# कर्नेल Origins

जो profile बताता है कि `r_128_3_32_4_2_2_2_4_4_192_2` ने 100 ms लिए, वह कर्नेल की shape बताता
है — यह नहीं कि कर्नेल किसका है। Origins इसी दूसरे सवाल का जवाब देते हैं: dispatch होने वाला हर
कर्नेल जानता है कि वह किस module path, call site या ONNX node के लिए बना था, और profiler समय को
उसी path के साथ rollup कर सकता है — per layer, per block, per stage।

यह पेज इस्तेमाल का guide है: इसे चालू कैसे करें, model को instrument कैसे करें और output कैसे
पढ़ें। Mechanism — हर node पर एक hash-consed field, जिसे kernel cut पर फिर से हटा दिया जाता है —
का सार अंत में है; पूरा ब्यौरा [IR design](./ir-design.md) और [op bestiary](./op-bestiary.md) पेजों पर है।

---

## इसे चालू करना {#turning-it-on}

Capture default में बंद रहता है, और बंद रहते हुए इसकी कोई क़ीमत नहीं: nodes कोई origin नहीं ढोते,
और hashes उस build के hashes से byte-identical रहते हैं जिसमें यह feature है ही नहीं। दो
switches:

| Switch | असर |
|--------|--------|
| `SVOD_ORIGIN=1` | पूरे process के हर thread पर capture चालू (unset, ख़ाली या `0` = बंद) |
| `SVOD_ORIGIN_DEPTH=<n>` | rollups पहले `n` path segments रखते हैं (unset या `0` = पूरा path, `depth leaf` के रूप में छपता है) |

```bash
SVOD_DEVICE=AMD:0 SVOD_ORIGIN=1 cargo run --release -p svod-model --example gigaam_infer -- \
    audio.wav --profile --origin-depth 3 --profile-json profile.json
```

`--origin-depth` को `SVOD_ORIGIN_DEPTH` पर प्राथमिकता मिलती है; Whisper example में सिर्फ़
`--profile` है।

Tests में capture सिर्फ़ मौजूदा thread पर चालू करें, ताकि साथ-साथ चलने वाले tests अपनी graph
identity बनाए रखें:

```rust
let _capture = svod_ir::origin::capture_for_thread(true); // restored on drop
```

---

## Origins आते कहाँ से हैं {#where-origins-come-from}

Origin frames का एक path है — root सबसे पहले। हर frame इनमें से कोई एक होता है:

| Frame | कैसे दिखता है | कौन खोलता है |
|-------|-------------|-----------|
| `Module` | `encoder.layers.3.ffn1` | model code, हर module पर एक segment (`OriginScope::module`) |
| `Label` | `vad`, `GigaAmCtcJit`, `initializer` | pipeline stages, हर `jit_wrapper!` build, ONNX importer |
| `Onnx` | `/encoder/Conv` या `#12:MatMul` | ONNX importer, हर node पर एक: node का नाम, या नाम न हो तो `#index:op_type` |
| `Call` | `@ linear model/src/gigaam/encoder.rs:43` | हर public `Tensor` op, अपने entry पर `origin_call!` के ज़रिए |

नाम वाले frames `.` से जुड़ते हैं; `Call` frame एक space के बाद आता है। यह module path के नीचे
बैठी flat file:line layer है: कोई op इसे तभी खोलता है जब मौजूदा frame पहले से call न हो (सबसे
बाहरी जीतता है), इसलिए दूसरी ops के ऊपर लिखी op (`matmul` के ऊपर `linear`) user की line एक ही
बार दर्ज करती है, svod का अपना source कभी नहीं। उसके ऊपर की module layers वही हैं जो model code
जोड़ता है।

### एक Rust model को instrument करना {#instrumenting-a-rust-model}

`forward` में हर module के लिए scope ठीक उसी तरह खोलें जैसे आप उसका state-dict prefix लिखते हैं।
Model crate में यही काम करने वाले helpers मौजूद हैं:

```rust
use crate::state::{scoped, scoped_index};

fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let x = scoped("subsampling", || self.subsampling.forward(x))?;
    let mut x = x;
    for (i, layer) in self.layers.iter().enumerate() {
        x = scoped_index("layers", i, || layer.forward(&x))?;   // layers.0, layers.1, …
    }
    scoped("final_norm", || self.final_norm.forward(&x))
}
```

`scoped` closure के चारों ओर `OriginScope::module(name)` खोलता है; `scoped_index` `name.i` segment
को सिर्फ़ तभी format करता है जब capture चालू हो; `scope_index(name, i)` वही काम guard के रूप में
करता है, उन loop bodies के लिए जो closures नहीं बन सकतीं। हर module सिर्फ़ अपना segment खोलता है;
nesting से पूरा path फिर बन जाता है, इसलिए profile में जो path छपता है वह उन weights के state-dict
key prefix के बराबर होता है जिन्हें उसने छुआ। GigaAM और Whisper इसी तरह instrument किए गए हैं, और
`model/src/test/unit/origin.rs` assert करता है कि paths के दोनों sets मेल खाते हैं।

Pipeline stages root segment हैं। GigaAM JIT build को module scope में लपेटता है, `arch` pipeline
label इस्तेमाल करती है — दोनों ही तरह stage का नाम अंदर बने हर path की शुरुआत में रहता है:

```rust
scoped("ctc_head", || jit.prepare_with_config(mel_spec, lengths_spec, &config))?;

let _stage = OriginScope::label(self.profile_label());   // "vad"
self.vad.probs(waveform)?;
```

`jit_wrapper!` macro हर build closure के चारों ओर `OriginScope::label("<WrapperName>")` खोलता है —
नीचे दिखने वाला `GigaAmCtcJit` segment यहीं से आता है। किसी भी scope के बाहर बनी हर चीज़
`<unattributed>` row में जा गिरती है।

### ONNX graphs {#onnx-graphs}

कुछ करने की ज़रूरत नहीं। Importer हर node के लिए एक `Onnx` frame खोलता है (index, name, op type,
domain, opset), और हर subgraph branch (`then_branch`, `else_branch`) के लिए एक `Label` — उसी node
के नीचे जिसका वह branch है, इसलिए किसी `If` की body `#7:If.then_branch.#0:Add` जैसी दिखती है।
Initializers और graph inputs `initializer` और `input` के नीचे बैठते हैं — लेकिन सिर्फ़ तब जब वे
कोई computed node बनाते हैं: literals और buffers कभी origin नहीं ढोते।

### हाथ से लिखे कर्नेल {#hand-written-kernels}

एक `tk` कर्नेल उसी scope के खाते में जाता है जो उसे बनाते समय सक्रिय था — वही नियम जो graph
कर्नेल पर लागू है। Scheduler उसकी body कभी नहीं देखता, इसलिए `UOp::custom_kernel` construction के
समय ही उसे harvest करके हटा देता है (`ir/src/uop/constructors/graph.rs`); एक ही हाथ के कर्नेल को
launch करने वाली दो layers अब भी एक ही compiled program साझा करती हैं। जिन inputs को यह copy
करता है, वे अपने producer का origin विरासत में पाते हैं।

---

## Output पढ़ना {#reading-the-output}

Capture चालू हो तो `--profile` वही आम per-kernel table print करता है और उसके बाद दो rollups। यह
sample GigaAM v3 encoder का है — f16, gfx1151 पर एक 60 s window, depth 3 पर काटा हुआ:

```
519 dispatches (519 GPU-stamped), total 444.237 ms
  total ms  count    mean µs      %  name
   103.183     16     6448.9   23.2  r_128_3_32_4_2_2_2_4_4_192_2n1
   100.305     16     6269.1   22.6  r_128_3_32_4_2_2_2_4_4_192_2
    80.530     32     2516.6   18.1  r_128_12_32_4_2_2_2_4_4_48_2
    …
origin rollup (depth 3, exclusive; rows sum to the total):
  total ms  count    mean µs      %  origin path
    27.833     32      869.8    6.3  ctc_head.GigaAmCtcJit.layers.3
    27.678     32      864.9    6.2  ctc_head.GigaAmCtcJit.layers.9
    27.620     32      863.1    6.2  ctc_head.GigaAmCtcJit.layers.0
    …
    23.334      2    11666.8    5.3  ctc_head.GigaAmCtcJit.subsampling
     0.661      4      165.2    0.1  ctc_head.GigaAmCtcJit.head
     0.131      1      131.0    0.0  ctc_head.GigaAmCtcJit
origin rollup (depth 3, inclusive; parents contain children, rows overlap):
  total ms  count    mean µs      %  origin path
   444.237    519      855.9  100.0  ctc_head
   444.237    519      855.9  100.0  ctc_head.GigaAmCtcJit
    27.833     32      869.8    6.3  ctc_head.GigaAmCtcJit.layers.3
    …
```

इसे कैसे पढ़ें:

- **Exclusive** हर dispatch को एक ही बार, उसके *primary* origin के खाते में डालता है: वह scope
  जिसने वह value बनाई जिसे कर्नेल store करता है (या, अगर उस value पर कोई origin नहीं, तो root के
  नीचे का सबसे नज़दीकी attributed node)। Rows कुल को बाँटती हैं, इसलिए सोलह `layers.N` rows,
  `subsampling`, `head` और बची हुई `GigaAmCtcJit` row मिलकर 519 dispatches और 444 ms बनाती हैं।
  सोलह layers × 32 dispatches पूरा encoder है; per-layer फैलाव (25.3 से 27.8 ms) असली है और
  सबसे पहले आप इसी को देखेंगे। बारह से कम exclusive rows हों तो हर row अपने top तीन कर्नेल भी
  `· name` lines के रूप में दिखाती है।
- **Inclusive** हर dispatch को उसमें fuse हुए हर origin के हर ancestor के खाते में डालता है।
  Parent row में उसके children शामिल होते हैं, इसलिए `ctc_head` 100 % है और rows overlap करती हैं।
  इससे देखें कि किसी block का कितना समय उन कर्नेलों में छिपा है जो module boundaries के पार fuse
  हुए।
- **Depth** रखे गए path segments की संख्या है। यहाँ depth 3 per-layer rows देती है; depth 4 एक
  layer को `ffn1`, `mhsa`, `conv`, `ffn2`, `final_norm` में बाँटती है; leaf पूरा path रखता है।
  `Call` frames कभी rollup keys नहीं बनते — वे कर्नेल rows और JSON में detail हैं।
- जो कर्नेल दो modules को fuse करता है, वह exclusive रूप से उसी के खाते में जाता है जिसकी value
  वह store करता है (residual add layer पर पड़ता है, `ffn2` पर नहीं), और inclusive रूप से दोनों के।

हर `RunProfile` दोनों rollups रखता है: `render_report()` (ऊपर वाला layout, जिसे `gigaam_infer`
इस्तेमाल करता है) और `render_table()` (Whisper का `--profile`, एक अलग कर्नेल table) वही origin
section जोड़ते हैं, उसी depth पर काटा हुआ जिस पर profile बना था; `render_report_at(d)` /
`render_table_at(d)` / `to_json_at(d)` इसे override करते हैं।

### JSON {#json}

`--profile-json out.json` (या `RunProfile::to_json()`) हर run के लिए एक document लिखता है
(संक्षिप्त):

```json
{
  "origin_depth": 3,
  "stages": [{
    "name": "ctc_head", "wall_ms": 463.8, "gpu_ms": 444.2, "dispatches": 519, "meta": {},
    "kernels": [{
      "name": "r_128_3_32_4_2_2_2_4_4_192_2", "count": 1, "total_ms": 6.3, "mean_us": 6269.1,
      "origin": "ctc_head.GigaAmCtcJit.layers.3 @ add model/src/gigaam/encoder.rs:596",
      "origin_id": 41, "origins": ["…"], "origin_ids": [41, 39]
    }],
    "origins_exclusive": [{ "path": "ctc_head.GigaAmCtcJit.layers.3", "count": 32, "total_ms": 27.8, "mean_us": 869.8, "percent": 6.3, "kernels": [] }],
    "origins_inclusive": []
  }],
  "origins": [{ "id": 41, "parent": 40, "frame": { "Module": { "name": "layers.3" } } }]
}
```

कर्नेल rows की key entry point *और* primary origin, दोनों मिलकर बनाते हैं, इसलिए एक ही program
हर उस scope के लिए एक बार दिखता है जिसने उसे dispatch किया। `origins` में सिर्फ़ वे frames होते
हैं जिनका run ने ज़िक्र किया, `parent` के तहत closed — इसलिए ids उस process के बिना भी resolve हो
जाती हैं जिसने file लिखी थी।

---

## Threads {#threads}

Capture state हर thread का अपना होता है: switch, मौजूदा scope, और यह कि वह scope call frame है
या नहीं। Scopes काम के पीछे-पीछे दूसरे threads पर नहीं जाते; scope guard `!Send` है और उसी
thread को restore करता है जिस पर वह खोला गया था। इससे निकलने वाले नियम:

- Graph उसी thread पर बनाएँ जिसने scopes खोले थे। GigaAM और Whisper यही करते हैं;
  `prepare_with_config` के चारों ओर खोला गया stage scope उसके अंदर बनी हर चीज़ को ढक लेता है।
- Scheduling और compiling detached चलते हैं (`OriginScope::suspend`) — caller पर भी और rayon
  workers पर भी — ताकि कोई ambient scope कभी kernel body में न रिसे; attribution तब तक CALL पर
  harvest हो चुका होता है।
- किसी ऐसे worker तक scope ले जाने के लिए जिसे आप ख़ुद spawn करते हैं, `origin::current()` (एक
  `Option<OriginId>`) capture करें और वहाँ `origin::install(id)` से फिर install करें। Workers किसी
  भी दूसरे thread की तरह अपना switch `SVOD_ORIGIN` से लेते हैं।
- BEAM search एक child process में origin-free kernel bodies पर चलता है; उसे कोई scope कभी नहीं
  दिखता।
- **Async code:** scopes का nest होना ज़रूरी है, इसलिए किसी scope को `.await` के पार न पकड़े रहें।
  Scope खोलें, graph synchronously बनाएँ, उसे drop करें, फिर await करें। Guard `!Send` है, इसलिए
  जो future किसी guard को await के पार ज़िंदा रखता है उसे multi-threaded executor पर spawn नहीं
  किया जा सकता, और debug build तब panic करता है जब दो tasks एक thread पर scopes को आपस में
  गूँथ दें (कोई guard तब drop हो जब बाद वाला अब भी सक्रिय हो)। svod में graph construction
  synchronous है, इसलिए code का स्वाभाविक ढाँचा पहले से ही इस शर्त को पूरा करता है।

---

## लागत और समझौते {#costs-and-trade-offs}

- **बंद:** कुछ नहीं। हर node पर एक thread-local read, कोई allocation नहीं, hashes अपरिवर्तित।
- **चालू:** हर scope entry पर एक interning (arena पर एक mutex, हर forward में सैकड़ों बार), call
  frame के लिए हर public op पर एक thread-local write, और union harvest करने के लिए cut पर हर
  कर्नेल का एक toposort। GigaAM के dispatch counts और GPU time capture चालू और बंद दोनों में एक
  जैसे हैं।
- **Identity बदल जाती है।** Origin किसी node की identity का हिस्सा है, इसलिए अलग-अलग scopes में
  बने दो एक जैसे expressions तब तक दो nodes रहते हैं जब तक cut उन्हें हटा न दे। Kernel programs
  पर असर नहीं — strip dedup लौटा देता है — लेकिन जो helper हर call site पर वही expression दोबारा
  बनाता है (mask clamp, table cast, input copy) वह उसे हर scope में एक बार materialise करेगा। ऐसे
  helpers को `OriginScope::suspend()` के नीचे चलाएँ, या copy को उसके producer का origin विरासत में
  लेने दें; `custom_kernel` अपने inputs के लिए पहले से यही करता है। Constants, buffers, params,
  `UNIQUE`, `DEFINE_VAR`, `BIND`, `STACK` और हर `Index`-typed node इसी कारण कभी origin नहीं ढोते।
- **जो tests structural identity पर टिके हैं** (हाथ से बने दो graphs जिनसे उम्मीद है कि वे
  hash-cons होकर एक node बनेंगे), उन्हें `capture_for_thread(false)` के साथ चलना चाहिए।

---

## यह काम कैसे करता है, एक पैराग्राफ़ में {#how-it-works-in-one-paragraph}

किसी scope के सक्रिय रहते बना हर `UOp` उस scope का 4-byte `OriginId` (एक `NonZeroU32`, इसलिए
`Option` भी चार bytes ही है) रखता है और उसे अपने content hash में मिला देता है, ताकि अलग-अलग
scopes के एक जैसे subgraphs rangeify के दौरान अलग बने रहें। Kernel cut पर `split_store` body पर
एक बार चलता है, stored value के origin को primary और union को set मानता है, दोनों को kernel
`CALL` के `CallInfo` पर दर्ज करता है, और body को origins हटाकर फिर से बनाता है। Cut के बाद सब कुछ
— optimizer, BEAM, codegen, हर kernel cache — origin-free ASTs देखता है। Plan CALL का attribution
हर prepared op पर copy करता है, profiler हर `KernelProfile` पर, और rollups parent chain को माँगी
गई depth तक काट देते हैं।
