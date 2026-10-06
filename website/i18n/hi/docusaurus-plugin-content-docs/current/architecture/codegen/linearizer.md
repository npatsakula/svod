---
sidebar_label: Late rewrites और linearizer
---

# Late Rewrites, प्रोग्राम सीमा और Linearizer (स्टेज 19–20 और आगे)

आख़िरी post-optimization स्टेज ग्राफ़ को एक ठोस backend के लिए render करने योग्य बनाते हैं: जो operations target में नहीं हैं उन्हें decompose किया जाता है, validity एक `gate` बन जाती है, weak dtypes commit किए जाते हैं। फिर `svod-codegen` control-flow edges जोड़ता है, parameters को क्रमांकित करता है और DAG को instruction सूची में समतल करता है। सोर्स: `optimizer/mod.rs`, `late/gater.rs`, `late/dtype.rs`, `optimizer/implicit_barriers.rs`, `linearize/`, `codegen/src/program_pipeline.rs`।

नीचे का हर matcher renderer की क्षमता तालिका (`renderer.supported_ops()`, `supports_dtype`) से बनता है, इसीलिए `optimize_kernel_with_config` ऐसे renderer को अस्वीकार करता है जिसके पास यह तालिका नहीं है (`OptError::MissingRendererCapabilities`)।

## 19 — float ALU operands का cast

`pm_cast_float_alu`: `Sin`, `Log2`, `Exp2`, `Sqrt`, `Reciprocal` के लिए operand को परिणाम dtype में cast करना। Transcendental decompositions dtype-समरूप polynomials में फैलते हैं और उन्हें मिश्रित-dtype operand नहीं दिखना चाहिए।

## 19b — early decompositions

`early_decomposition_patterns(supported_ops)`:

```text
symbolic_simple + pm_fold_cast_const + pm_mod_to_and + divmod_decomposition_patterns
  + pm_threefry_decomp        if !supports(Threefry)
  + pm_max_decomposition      if !supports(Max) && supports(Lt)
  + pm_erf_decomposition      if !supports(Erf)
```

`divmod_decomposition_patterns` (`ir/src/decompositions/mod.rs`) floor division और modulo (`FloorDiv`/`FloorMod`) को sign सुधार के साथ truncating `CDiv`/`CMod` में lower करता है — वह रूप जो हर backend के पास है। `pm_mod_to_and` यहाँ भी है और late सेट में भी, ताकि दो की घात वाले modulo truncating lowering के देखने से पहले fold हो जाएँ।

## 19c — dtype decompositions

`DTypeDecompCtx` के साथ `pm_dtype_decomp_commit = pm_dtype_decomps + pm_commit_weak`। पहला नियम केवल यह दर्ज करता है कि `FP8E4M3`, `FP8E4M3FNUZ`, `FP8E5M2`, `FP8E5M2FNUZ`, `Float16`, `BFloat16`, `Int64`/`UInt64` में से कौन से ग्राफ़ में आते हैं; फिर `SINK` नियम bottom-up और dtype क्रम में हर उस दर्ज dtype को फिर से लिखता है जिसे renderer समर्थन नहीं देता:

| असमर्थित | किस रूप में emulate | Matcher |
|-------------|-------------|---------|
| `Int64`, `UInt64` | दो `Int32`/`UInt32` words: `PARAM`/`BUFFER` का आकार दोगुना, `INDEX` पर word का tag, `Lt` से बने carry और borrow, `CDiv`/`CMod` के लिए 64-चरण shift-subtract divider | `pm_long_decomp` (`devectorize.rs`) |
| FP8 | समर्थित हो तो `Float16`, अन्यथा `Float32`; storage 8-bit unsigned word रहता है, `f2f` bit-सटीक रूपांतरण करता है (RNE rounding, FNUZ NaN encoding, `f2f_clamp` में saturation) | `pm_float_decomp` |
| `Float16`, `BFloat16` | `Float32` गणना, वही `f2f` storage रूपांतरण | `pm_float_decomp` |

`get_dtype_decomps` renderer की compile cache key के लिए वही चयन एक सूची के रूप में लौटाता है।

## 19d — late decompositions

`pm_decomp = early_decomposition_patterns + get_late_rewrite_patterns(renderer, disable_fast_idiv) + get_transcendental_patterns(supported_ops, TRANSCENDENTAL >= 2) (+ renderer.decomposition_matcher())`, fixpoint तक चलाया जाता है। Late सेट क्षमता-आधारित है ([strength reduction](../optimizations/strength-reduction.md) में हर नियम है):

```text
pm_mod_to_and + pm_half_bf16_cast                       always
+ pm_demorgan                                           if supports(Or)
+ pm_mul_to_shl                                         if supports(Shl)
+ pm_div_to_shr                                         if supports(Shr)
  + fast_division_patterns + pm_mod_to_idiv             if supports(Shr) && DISABLE_FAST_IDIV=0
+ pm_neg_from_mul                                       if supports(Neg)
+ pm_comparison_negations                               if supports(Lt) || supports(Eq)
+ pm_fma_decomposition                                  if supports(MulAcc)
  + pm_shl_add_to_mulacc                                if supports(MulAcc) && supports(Shl)
+ pm_fdiv_to_mul                                        if supports(Fdiv)
```

`get_transcendental_patterns` (`ir/src/decompositions/`) f16/f32/f64 के लिए `Exp2`, `Log2`, `Sin` को `xexp2`/`xlog2`/`xsin` polynomial सन्निकटनों से बदलता है (अन्य floats f32 में गणना करते हैं) और `Sqrt` को `xpow(x, 0.5)` से — हर उस op के लिए जो renderer में नहीं है, या `TRANSCENDENTAL=2` होने पर सभी के लिए। `decomposition_matcher` डिवाइस के `Renderer::decompositor()` hook की ऑप्टिमाइज़र-पक्ष वाली प्रति है; Metal वहाँ `amd_decomposition_patterns` स्थापित करता है (native `exp2`/`log2` के ऊपर `Exp`, `Log`, `Cos`, `Tan` और binary `Pow`)।

उदाहरण में यह स्टेज `R1 * 64` को `R1 << 6` में और `R0 * 4 + (R1 << 6)` को `MulAcc(R0, 4, R1 << 6)` में बदलता है — `pm_shl_add_to_mulacc` द्वारा बनाया गया integer FMA।

## 19e — gates, register lanes, float demotion

`pm_move_gates_from_index` (`late/gater.rs`, Tinygrad के `gater.py` का port) अंततः validity को index से बाहर ले जाता है:

| पहले | बाद में |
|--------|-------|
| `LOAD(INDEX(buf, WHERE(g, idx, Invalid)))` (कोई `alt` नहीं, कोई `gate` नहीं) | `LOAD { index: INDEX(buf, idx), alt: 0, gate: g }` |
| `STORE(INDEX(buf, WHERE(g, idx, Invalid)), v)` (कोई `gate` नहीं) | `STORE { index: INDEX(buf, idx), value: v, gate: g }` |
| `SHRINK` पर वही दो रूप (coalesced समूह) | साफ़ किए गए `SHRINK` पर gated `LOAD`/`STORE` |
| एक साझा शर्त वाला image दो-coordinate `INDEX` | एक gated एक्सेस (पहले जाँचा जाता है) |
| `WHERE(g, LOAD{gate: g}, alt)` और उल्टा रूप | `alt` load में fold हो जाता है |

`valid_index` को तीसरे `WHERE` स्लॉट में शाब्दिक `Invalid` constant चाहिए। फिर `pm_scalarize_register_stack_index_preserve_deps` `INDEX(AFTER(STACK(..), deps), c)` — कुछ stores के बाद पढ़ा गया register stack का एक lane — को चुने गए `LOAD` में बदलता है, जिसके address पर deps फिर से जोड़े जाते हैं, और `merge_register_read_ends` उन `END`s को मिलाता है जो एक ही register `AFTER` के तहत समान ranges बंद करते हैं (एक debug assertion जाँचता है कि कोई register-stack `INDEX` नहीं बचा)। `demote_unsupported_floats` (`late/dtype.rs`) सबसे अंत में चलता है: `Float64` ALU रहित renderer (Metal, WebGPU) पर हर आंतरिक f64 मान f32 में गणना होता है, जबकि global f64 storage, उसके loads और उनके `alt` मान चौड़ा dtype बनाए रखते हैं।

## 20 — final rewrite

```text
pm_final = pm_commit_weak + pm_cast_weak + pm_decomp (+ renderer.extra_matcher()) + pm_split_ends
```

एक fixpoint (debug builds में `assert_target_renderer_boundary` पहले चलता है: कोई स्थिर multi-index `INDEX` नहीं, कोई बचा हुआ singleton broadcast नहीं, कोई मिश्रित `STACK`/vector ALU नहीं)। `pm_split_ends` `END(x, [r1, r2, r3])` को `END(END(END(x, r3), r2), r1)` में बदलता है, ranges `(axis_id, axis_type.priority())` के घटते क्रम में; `Void`/`Bool` sources (reduction backedges) अलग किए जाते हैं और सबसे बाहरी `END` पर फिर से जोड़े जाते हैं, और मूल tag बनाए रखा जाता है ताकि बाद के merge चरण उसे फिर भी ढूँढ सकें। `extra_matcher` `svod_device::device::Renderer` पर प्रति-backend hook है; यह decompositions वाले उसी fixpoint के अंदर चलता है। CPU और NVPTX renderers फिर से `bool_storage_patterns` स्थापित करते हैं (`cpu_extra_matcher`), AMD `amd_non_native_fp8_patterns` स्थापित करता है (OCP FP8 ALU को f32 तक चौड़ा किया जाता है; storage, रूपांतरण और MFMA operands अछूते रहते हैं)।

फिर, अलग पासों के रूप में: `pm_remove_invalid` बचे हुए हर data-typed `WHERE(c, x, Invalid)` को `WHERE(c, x, 0)` से और हर `Invalid` `STACK` lane को शून्य से बदलता है (एक debug assertion जाँचता है कि कोई नहीं बचा), और `add_implicit_barriers` local मेमोरी के लिए `BARRIER`s डालता है: local buffer पर ऐसे `AFTER` से पहले एक RAW barrier जिसके deps में बिना barrier वाला local `STORE` हो, और ऐसे लूप body के अंत में एक WAR barrier जो उस local buffer में store करता है जिसे उसी लूप का कोई अन्य load पढ़ता है। `optimize_kernel_with_config_and_final_rewrite` parity टूलिंग के लिए barriers से ठीक पहले capture किया गया ग्राफ़ लौटाता है। `graph_rewrite` द्वारा गिराया गया `KernelInfo` metadata फिर से जोड़ा जाता है।

## प्रोग्राम सीमा (`program_from_sink`)

`svod-codegen` कमान सँभालता है (`program_pipeline.rs`):

1. **`add_control_flow`** (`linearize/mod.rs`): फिर से `pm_split_ends` (idempotent), फिर `CFGContext::new(sink)` और bottom-up `pm_add_control_flow`। Context हर `END` के लिए गणना करता है कि वह किस `END`/`SINK` में नेस्टेड है — `END x` `u` में नेस्टेड है जब `u` `x` पर निर्भर हो और `u` का range `x` की निर्भरताओं में हो — siblings को parent के अनुसार समूहित करता है, उन्हें इस आधार पर क्रमित करता है कि वे कितने siblings पर निर्भर हैं, और हर बाद वाले sibling के `RANGE` से उसके पूर्ववर्ती तक एक edge दर्ज करता है (पिछले sibling का `END`, या पहले वाले के लिए parent का `RANGE`)। `pm_add_control_flow` पूर्ववर्ती को `RANGE` के sources में जोड़ता है; तब `InScopeRangesProperty` nesting देखता है, और यही नीचे नेस्टेड ranges को बड़ा `run_count` देता है। ऐसा पूर्ववर्ती जिसमें पहले से range हो, panic करता है (`"edge would create cycle"`)।
2. **`number_params`** अंतिम `PARAM` स्लॉट असाइन करता है (`validate_param_slots` असाइन न हुए या दोहराए गए स्लॉट को अस्वीकार करता है)।
3. `SVOD_SPEC` चालू होने पर `spec_program` के विरुद्ध **`verify_final_sink`**; `ProgramInfo::from_sink` ABI पढ़ता है।
4. **`pre_isel_matcher` / `isel_matcher`** — `svod_device::device::Renderer` पर दो instruction-selection hooks, दोनों bottom-up (`PreIselContext`, `IselContext`)। ये ISA-स्तर के backends के लिए हैं; LLVM और C renderers इन्हें `None` छोड़ते हैं।
5. `UOp::program(sink, info, None, None, None)` — `PROGRAM` नोड, जिसके बाद वाले sources `LINEAR`, `SOURCE` और `ProgramBinary` स्टेज हैं।

## `linearize`

`do_linearize` `svod_schedule::linearize(sink)` (`linearize/linearize.rs`, Tinygrad के `linearizer.py` का सीधा port) को बुलाता है, फिर `line_rewrite_cleanups`, फिर `spec_program` के विरुद्ध `verify_linear_list`।

हर नोड की sort key `(run_count, priority, extra, tuplize rank)` है:

| Op | प्राथमिकता |
|----|----------|
| `PARAM` | −20, बराबरी स्लॉट (`extra`) से तय |
| `BUFFER` (global, register) | −18 |
| `BUFFER` (`AddrSpace::Local`) | −17 |
| `END` | −5 |
| `LOAD` | −1 |
| बाकी सब (`CONST`, ALU, `SPECIAL`, …) | 0 |
| `STORE` | +1 |
| `RANGE` | +5 |

`run_count = prod(vmax + 1)` नोड के scope वाले ranges पर (symbolic extent 1 गिना जाता है), इसलिए लूप के बाहर का कोड लूप body से पहले आता है। Tuplize rank Tinygrad की `(op, arg, dtype, *src.tuplize)` key है, जो toposort (`TuplizeKeys`) पर पुनरावृत्त रूप से गणना की जाती है, जिससे क्रम पूर्ण और नियतात्मक बनता है। फिर इन ranks पर `SINK` से max-heap toposort द्वारा रैखिक सूची बनाई जाती है, जिसे अंत में उलट दिया जाता है: कोई नोड तब निकलता है जब उसके सभी उपभोक्ता निकल चुके हों, इसलिए परिभाषाएँ पहले आती हैं, `LOAD`s अपने उपयोगों से पहले, `STORE`s गणना के बाद, और हर `RANGE` अपनी body से ठीक पहले खुलता है।

`SVOD_DUMP_LINEAR=<dir>` scope वाले range ids के साथ toposort (`tree_<id>.txt`) और अंतिम सूची (`linear_<id>.txt`) लिखता है।

## `line_rewrite_cleanups`

`line_rewrite` instruction सूची पर एक बार चलता है; हर प्रविष्टि को कई से बदला जा सकता है। एकमात्र cleanup `linearize_cleanup_pattern` है: `Bool` gate वाला ऐसा `STORE` जिसका address `INDEX`/`SHRINK` है (संभवतः किसी `CAST` के पीछे), `IF(gate)`, बिना gate वाला `STORE`, `ENDIF` बन जाता है। `IF` और `ENDIF` दोनों केवल सूची में मौजूद होते हैं, ग्राफ़ में कभी नहीं (`"if not allowed in graph"`), इसीलिए `spec_program` को sink के साथ-साथ सूची पर भी जाँचा जाता है। जो backends store को predicate कर सकते हैं (LLVM, CUDA, Metal) वे इस तिकड़ी को conditional store के रूप में render करते हैं।

यह सूची `PROGRAM` का `LINEAR` स्टेज है; `do_render` इसे `Renderer::render` को सौंपता है, और `do_compile` binary बनाता है। [उदाहरण सहित विवरण](./worked-example.md) row-sum kernel के लिए CPU renderer द्वारा उत्पन्न LLVM IR दिखाता है।
