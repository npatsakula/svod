---
sidebar_label: डीबगिंग
---

# कर्नेल को Debug और Verify करना

एक हाथ से लिखा कर्नेल उतना ही भरोसेमंद होता है जितनी उसे जाँच पाने की आपकी क़ाबिलियत।
[Flash Attention](./flash-attention) walkthrough ने दिखाया कि किस तरह का कर्नेल हाथ से लिखने लायक़ होता है;
यह chapter वह है जिससे आप उस पर भरोसा करना सीखते हैं। USE चेहरा आपको एक lazy `Tensor` देता है जो एक बड़े
graph में fuse हो जाता है — सुविधाजनक तो है, पर "क्या यह एक कर्नेल correct है, और यह कितना तेज़ है?" यह
पूछने के लिए बुरी जगह। `tk` का **DEBUG चेहरा** ठीक इसी के लिए है: एक अकेले कर्नेल को concrete buffers के
ख़िलाफ़ run करो, नतीजा वापस पढ़ो, उसे time करो, और साबित करो कि किसी refactor ने इसका behavior नहीं बदला।

---

## Direct dispatch: एक कर्नेल run करो, bytes देखो

direct-launch API (`tk/src/launch.rs`) tensor scheduler को पूरी तरह bypass कर देता है। आप इसे एक कर्नेल
body और असली input tensors देते हैं; यह inputs को realize करता है, outputs allocate करता है, render, compile
और dispatch करता है, और नतीजा एक ऐसे output में लिख देता है जिसे आप वापस पढ़ सकते हैं:

```rust
// The DEBUG face from tk/src/lib.rs. `outs` are written in place.
run_kernel("tile_add", [1, 1, 1], block, &mut [&mut out], &[&input_a, &input_b], build)?;
let values = out.as_vec::<f32>()?;   // read the GPU result straight back
assert_eq!(values, expected);
```

चूँकि यह scheduling, fusion, और dependency tracking को छोड़ देता है, इसलिए आप जो measure करते हैं वह *सिर्फ़
आपका कर्नेल* होता है — कोई ऐसा graph नहीं जिसमें यह बस शामिल हो। यही isolation असल मुद्दा है: जब कोई number
ग़लत निकले, तो आप जानना चाहते हैं कि वह *यहीं* ग़लत है, न कि किसी fused pipeline में कहीं और।

path पर एक छोटी-सी बात: *scheduler* को छोड़ देना *optimizer* को छोड़ देना नहीं है। `compile` आपके `SINK` पर
production वाला `optimize_kernel_with_config` अब भी चलाता है — जो हाथ से lower किए गए body पर शून्य schedule
opts apply करता है (यही `opts_to_apply: Some(vec![])` marker ख़रीदता है), पर render से पहले हर कर्नेल को
ज़रूरी वे साझा rewrites अब भी करता है, जिनमें index-dtype lowering भी है। scheduler के बिना भी आपको correct
code मिलता है। `ArchCaps` buffers के device से आते हैं; जिस GPU का arch resolve नहीं होता वह एक error है, और
सिर्फ़ host device ही `ArchCaps::GFX942` पर fallback करता है ताकि `SINK` फिर भी build हो जाए।

---

## असली hardware पर timing

performance वाले काम के लिए, `CompiledLaunch` (`compile_kernel` से) wall-clock अंदाज़ों के बजाय hardware
timestamps expose करता है:

```rust
// Render + compile once …
let launch = compile_kernel("matmul", grid, block, &mut [&mut c], &[&a, &b], build)?;
// … then dispatch in a loop, outside the timed region.
// SAFETY: the bound buffers stay allocated for `launch`'s lifetime.
unsafe { launch.dispatch(true) }?;
let ns = launch.dispatch_gpu_ns()?;   // Option<u64>: device-measured dispatch time
```

`dispatch_gpu_ns()` एक profiling context से एक बार dispatch करता है और उसके इर्द-गिर्द device के अपने
timestamp counters पढ़ता है, इसलिए आप device पर बीता समय measure कर रहे होते हैं, न कि इसे launch करने की
round-trip latency — जो backend कुछ stamp नहीं करता, उस पर यह `None` है। यही वह primitive है जिससे
[autotuner](./tuning) candidates को rank करता है, उसी `warm_clock` से clock उठाने के बाद जिसे benches
इस्तेमाल करते हैं। criterion benches वही stamps एक layer ऊपर, `plan.profile` के ज़रिए पाते हैं; देखें
[Profiling और Benchmarking](./profiling)।

---

## बिना GPU वाले tests, GPU वाले tests

`SINK` बनाना शुद्ध UOp construction है और इसे किसी device की ज़रूरत नहीं; सिर्फ़ उसे execute करने को है।
test module (`tk/src/test/unit/`) यह बँटवारा हर जगह इस्तेमाल करता है, और नए कर्नेल को भी करना चाहिए:

- **Graph-shape tests** हर `cargo test` पर चलते हैं। कर्नेल को placeholder buffers
  (`UOp::new_buffer(DeviceSpec::Cpu, size, dtype)`, `ArchCaps::GFX942`) के ख़िलाफ़ build करें, `SINK` को
  toposort करें, और assert करें कि उसमें क्या है और क्या नहीं — `guide.rs` जाँचता है कि tile-add कर्नेल एक
  `Op::Special` बनाता है, उसमें एक `Binary(Add)` है, और कोई `Wmma` और कोई `Local` buffer नहीं है।
- **Hardware tests** `#[ignore]` हैं और unsupported device पर ख़ुद skip हो जाते हैं:

```bash
SVOD_DEVICE=AMD:0  cargo test -p svod-tk --lib guide::test_tile_add_amd -- --ignored
SVOD_DEVICE=CUDA:0 cargo test -p svod-tk --lib fa::test_fa_graph_check -- --ignored --nocapture
```

Gates `tk/src/test/unit/mod.rs` में हैं: किसी कर्नेल के `ArchSet` के लिए `device_supported(archs)`, matrix-core
layouts चाहने वाली किसी भी चीज़ के लिए `fragment_device()`, layout-specific जाँचों के लिए `is_cdna_device()`
और `wave32_fragment_device()`। इनमें से हर एक `svod_tk::tune::set_enabled(false)` भी call करता है, ताकि कोई
numerics test हर छुए गए shape को tune न करे।

graph-native कर्नेल के लिए, `svod_tensor::custom_kernel_check!` पूरी तुलना generate कर देता है: एक shape और
dtype के random inputs, test होने वाला कर्नेल, एक reference closure, दोनों f32 में cast करके
`atol = rtol = tol` पर compare।

```rust
svod_tensor::custom_kernel_check! {
    test_fa_graph_check,
    inputs (q, k, v): shape [1, 128, 2, 64], dtype svod_dtype::DType::BFloat16,
    run: |q, k, v| {
        let out = crate::kernels::fa::flash_attention(q, k, v).expect("FA build");
        Ok::<_, crate::LaunchError>(out.expect("the FA kernel applies to [1, 128, 2, 64] bf16 on every supported arch"))
    },
    reference: fa_causal_reference,
    tol: 2e-2,
}
```

यहाँ decline (`Ok(None)`) reference की ख़ुद से तुलना करने के बजाय ज़ोर से fail होता है।

---

## Fingerprints: साबित करना कि एक refactor behavior-preserving है

हाथ से लिखे कर्नेल के साथ एक बारीक जोखिम रहता है: आप builder code "साफ़-सुथरा कर देते हैं", कर्नेल फिर भी
compile हो जाता है और वाजिब-से numbers भी देता है, पर *generated IR* किसी ऐसे तरीक़े से बदल जाता है जो बाद में
किसी ख़ास shape या किसी ख़ास architecture पर ही सामने आता है।

`KernelFingerprint` (`tk/src/fingerprint.rs`) इसी के ख़िलाफ़ guard करता है। LLVM render एक run से दूसरे run
तक deterministic नहीं है (node ids SSA names में रिस जाते हैं), पर *graph* है: हर UOp एक recursive structural
`content_hash` रखता है, और fingerprint `SINK` का वही hash है, बगल में node tags के एक order-independent fold
के साथ — एक `u128` `digest`, और पढ़ने लायक़ diff के लिए `op_counts` और `node_count`।

```rust
let fp = kernel_fingerprint(&sink);
assert_eq!(fp.digest, GOLDEN_MATMUL_DIGEST);  // structure unchanged ⇒ behavior unchanged
```

अगर fingerprint हिल जाए, तो आपने emitted IR बदल दिया — चाहे जान-बूझकर या नहीं — और golden test आपको इसकी
ओर देखने पर मजबूर कर देता है। `tk/src/test/unit/golden.rs` इसी तरह matmul और flash-attention builders
(causal, non-causal, masked) को lock करता है; failure paste करने के लिए नया digest print करता है, और जान-बूझकर
किया गया re-baseline दोनों graphs को dump और diff करके साबित किया जाता है। वही digests
[autotuner](./tuning) के on-disk store की key हैं, इसलिए कर्नेल में बदलाव उसकी tiles को दोबारा measure करवाता है।

---

## वे ग़लतियाँ जो error नहीं देतीं

एक tile कर्नेल एक dependency graph है, और एक छूटा हुआ edge ग़लत जवाब है, compile error नहीं। जो test suite
ने पकड़ी हैं:

| लक्षण | कारण | Fix |
|---|---|---|
| एक accumulator loop trips के पार पुरानी state ढोता है | एक per-trip re-init (`g.zero(acc)`) जिसकी loop counter पर कोई dependency नहीं, loop के ऊपर hoist हो जाता है | `g.zero(lp.reinit(acc))` |
| एक loop-carried tile loop के बाद pre-loop value पढ़ता है | अंतिम read loop के `END` के बाद ordered नहीं है | `acc.after(&lp.close())` या `lp.close_carry(acc)` |
| `finish` debug-assert करता है, या linearizer किसी loop का scope ग़लत लगाता है | दो stores एक ही `RANGE` को `END` करते हैं | हर loop के लिए एक closing store; बाक़ियों को उसमें chain करें |
| ग़लत buffers, पर गिनती सही | `gl` / `bind_abi` का क्रम launch के `[outs..., ins...]` से अलग है | पहले outputs declare करें, फिर inputs launch के क्रम में, optional buffers अंत में |
| एक कर्नेल चुपचाप ग़लत K/V stream पढ़ता है | `k`/`v` `q` के shape पर, उसी width के किसी अलग dtype के साथ bind हुए (`Kernel::gl` सिर्फ़ byte width जाँचता है) | dtypes को `validate` में validate करें, जैसा `flash_attention_with` करता है |
| CDNA पर correct, RDNA पर कचरा | कोई lane count या fragment constant hardcode किया गया | `caps.wave_size` और `ker.frag(role)` पढ़ें ([Layouts और wave size](./wave-portability)) |

जो error *देती* हैं, वे builder के asserts हैं: ऐसा tile dimension जो अपने fragment का multiple न हो, ऐसा
`k_step` जो matrix core के K edge का multiple न हो, ऐसा block जो पूरी waves न हो, single-wave op call करता
multi-wave group, बिना layouts वाले arch पर `ker.frag`। हर एक ग़लत value का नाम बताता है।

---

## किस सवाल के लिए कौन-सा tool

| आप क्या पूछ रहे हैं… | इस्तेमाल करें |
|----------------|-----|
| "क्या यह builder अब भी वही emit करता है जो मैं सोचता हूँ?" | `SINK` के toposort पर एक graph-shape test |
| "क्या यह कर्नेल सही numbers देता है?" | `run_kernel` + `as_vec`, या एक reference के ख़िलाफ़ `custom_kernel_check!` |
| "यह इस GPU पर कितना तेज़ है?" | `compile_kernel` + `dispatch_gpu_ns` |
| "क्या मेरे refactor ने emitted IR बदला?" | `KernelFingerprint` golden test |
| "tuner ने कौन-सी tile चुनी, और क्यों?" | `SVOD_TK_TUNE_DIR` के नीचे store file ([Autotuning](./tuning)) |
| "कहीं *device/driver layer* ही तो गड़बड़ नहीं कर रहा?" | [AMD Backend → Debugging](../backends/amd/debugging), [CUDA Backend → Debugging](../backends/cuda/debugging) |

वह आख़िरी row मायने रखती है: यह chapter *कर्नेल* को debug करने के बारे में है — वह IR जो आपने author किया और
वे numbers जो यह देता है। जब समस्या उससे नीचे की हो — queue dispatch, memory faults, driver, PTX JIT — तो
per-backend chapters सही जगह हैं:
[AMD](../backends/amd/debugging) और [CUDA](../backends/cuda/debugging)।

---

## यह क्यों ज़रूरी है

हाथ से authoring में आप optimizer की safety net छोड़कर control हाथ में लेते हैं। DEBUG चेहरा वही है जिससे आप
यह सौदा safely करते हैं: correctness bugs को localize करने के लिए isolation, ऐसे performance claims करने के
लिए hardware timestamps जिनका आप बचाव कर सकें, और structural fingerprints — ताकि "मैंने तो बस code साफ़ किया
था" चुपचाप "मैंने कर्नेल बदल दिया" में न बदल जाए। इन तीनों के साथ, एक हाथ से लिखा कर्नेल एक autotuned कर्नेल
जितना ही verifiable है।
