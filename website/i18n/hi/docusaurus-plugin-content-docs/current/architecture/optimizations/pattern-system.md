---
sidebar_label: पैटर्न इंजन
sidebar_position: 0
---

# पैटर्न इंजन {#the-pattern-engine}

Svod का लगभग हर पास `patterns!` macro से बने मैचर पर एक `graph_rewrite` है: rangeify के चरण, सिम्बॉलिक सरलीकारक, expander, devectorizer, decompositions, gate movement। अपवाद कुछ साधारण graph walks हैं (`memory_coalescing`, `merge_register_read_ends`, `linearize`) और line rewrite, जो रैखिक instruction सूची पर चलता है। यह पृष्ठ macro और इंजन का संदर्भ है। स्रोत: `macros/src/patterns/`, `ir/src/pattern/`, `ir/src/rewrite/engine.rs`; Tinygrad में इनके समकक्ष `tinygrad/uop/ops.py` में `UPat` और `graph_rewrite` हैं।

## `patterns!` DSL {#the-patterns-dsl}

एक ब्लॉक `pattern [if guard] => body` नियमों की सूची है, जिसके पहले वैकल्पिक रूप से `@context Type;` हो सकता है। बायाँ पक्ष Rust pattern syntax है, जिसे उन चीज़ों के लिए विस्तारित किया गया है जो Rust pattern `Arc<UOp>` किनारे के पार व्यक्त नहीं कर सकता। `schedule/src/symbolic/patterns.rs` से वास्तविक नियम:

```rust
// constant folding over thirteen binary ops, one rule body
for op in binary [Add, Mul, Sub, FloorMod, Max, Pow, FloorDiv, Fdiv, And, Or, Xor, Shl, Shr] {
    op(a @const(a_val), _b @const(b_val))
      => eval_binary_op(op, a_val, b_val).and_then(|r| folded_const(a.dtype(), r)),
},

// commutative identity with a guard; `name @ @zero` binds the constant node too
Add[x, zero @ @zero]
    if !x.dtype().is_float()
        || matches!(zero.op(), Op::Const(ConstValueHash(ConstValue::Float(v))) if v.is_sign_negative())
    => x.clone(),
Mul[x, @one] => x.clone(),

// a repeated name means the same node (Arc::ptr_eq), here across a struct field
original @ FloorDiv(x, x) => exact_integer_rewrite(original, 1.into_uop(x.dtype())),
original @ FloorMod(range @ Range { end, .. }, end) => exact_integer_rewrite(original, range.clone()),

// struct ops match by field; `..` skips the rest
Cast { src: Cast { src: x, dtype: intermediate }, dtype: outer }
    if x.dtype() == *outer && can_safe_cast(outer, intermediate)
    => x.clone(),
```

`schedule/src/expand.rs` से एक stateful मैचर:

```rust
crate::cached_patterns! {
    @context RangeMap;
    reduce @ Reduce { .. } => expand_reduce(reduce),
    range @ Range { end: _, axis_id, axis_type }
        if matches!(axis_type, AxisType::Upcast | AxisType::Unroll) && ctx.contains_key(axis_id)
        => expand_range(ctx, range),
    Wmma { a, b, c, metadata } if metadata.upcast_axes.is_some() => expand_wmma(ctx, a, b, c, metadata),
}
```

| रूप | अर्थ |
|------|---------|
| `Add(x, y)` | प्रकार के अनुसार ALU op, स्थितीय, क्रमबद्ध। नाम `svod_ir::op::alu` से resolve होते हैं, इसलिए गलत नाम या arity compile error है। `UnaryOp`, `BinaryOp` (`Add, Mul, Sub, FloorMod, CMod, Max, Pow, FloorDiv, CDiv, Fdiv, Lt, Le, Eq, Ne, Gt, Ge, And, Or, Xor, Shl, Shr, Threefry`), `TernaryOp` (`Where`, `MulAcc`)। |
| `Add[x, y]` | क्रमविनिमेय: ठीक दो children, दोनों क्रम आज़माए जाते हैं; guard और body एक बार उत्पन्न होते हैं और हर क्रम के लिए दोबारा आज़माए जाते हैं। |
| `Cast { src: x, dtype }` | field के अनुसार struct op। एक field child pattern है जब वह `_`, `@..`, snake_case नाम, या `(..)`/`[..]`/`{..}`/`@` पर लागू identifier हो; बाकी सब शब्दशः Rust pattern है (`axis_type: AxisType::Upcast`, `index: 2`)। `Some(pat)`/`None` `Option<Arc<UOp>>` children से मेल खाते हैं (`Load { alt: None, gate: Some(g), .. }`)। Unit ops बिना कोष्ठक के लिखे जाते हैं (`Noop`)। |
| `x` / `_` / `name @ pattern` | नोड को bind करना / अनदेखा करना / पूरे sub-match को bind करना |
| `c @const(v)` | एक `CONST` नोड और उसका `ConstValue` bind करना |
| `c @vconst(vs)` / `c @anyconst(vs)` | `VCONST` lanes / `CONST` या `VCONST` को `Vec<ConstValue>` के रूप में |
| `Const(<rust pattern>)` | `ConstValue` पर एक Rust pattern |
| `@zero` / `@one` | किसी भी संख्यात्मक dtype का scalar `CONST` 0 / 1 (`is_zero` `-0.0` और `false` से भी मेल खाता है) |
| दोहराया गया नाम | एक ही नोड (`Arc::ptr_eq`); दोहराया गया `@const` value नाम मानों की तुलना करता है |
| `for op in binary [A, B]` / `[*]` | कई ops (या किसी प्रकार के सभी) के लिए एक नियम body; `op` runtime op मान है, guard और body में उपयोगी |
| `pat if guard => body` | guard हर binding और `ctx` देखता है |
| `=> body` | `Arc<UOp>`, `Option<Arc<UOp>>` (`None` अस्वीकार करता है) या `RewriteResult`; `?` काम करता है, अकेली binding एक clone लौटाती है |
| `@context Type;` | पहला आइटम; closure को `ctx: &mut Type` मिलता है |

`cached_patterns!` का व्याकरण वही है और यह `LazyLock` से `&'static TypedPatternMatcher<C>` लौटाता है; `patterns!` एक नया मैचर बनाता है। दोनों `svod_schedule` से re-export होते हैं।

### Macro क्या उत्पन्न करता है {#what-the-macro-generates}

`Op` पर `#[op_enum]`/`PatternEnum` लगा है, जो `svod_ir::op::pattern_derived::OpKey` उत्पन्न करता है — हर op प्रकार के लिए एक सघन index, समूहित `Unary`/`Binary`/`Ternary` के लिए हर sub-op का एक slot — और `OpMask`। एक `patterns!` ब्लॉक **एक closure** में compile होता है, जो `SimplifiedPatternMatcher::add_block` से पंजीकृत होता है, साथ में हर नियम के लिए `(root mask, early-reject mask)` की एक स्थिर तालिका:

- एक ही स्थिर root प्रकार वाले लगातार नियम एक `match __key { __KEY_Add => { .. } .. }` साझा करते हैं; किसी arm के भीतर नियम स्रोत क्रम बनाए रखते हैं।
- बिना स्थिर root वाले नियम — wildcards (`x if ..`), `for` ब्लॉक, `@anyconst` roots — उन `match`es के *बीच* क्रमिक चरणों के रूप में उत्पन्न होते हैं, इसलिए प्राथमिकता शुद्ध स्रोत क्रम है, न कि "पहले indexed, अंत में wildcards"।
- हर नियम एक early-reject परीक्षण से शुरू होता है: उसकी निश्चित child स्थितियों को जिन op प्रकारों की ज़रूरत है, वे एक bit mask हैं जिसे root के `src_ops` के विरुद्ध जाँचा जाता है (Tinygrad का `UPat.early_reject`)।
- क्रमविनिमेय स्थान आलसी रूप से जुड़े candidate iterators बन जाते हैं; नेस्टेड क्रमविनिमेय नोड नेस्टेड loops बनते हैं; body हर क्रम के लिए दोबारा आज़माई जाती है।
- एक `for` ब्लॉक हर नियम body के लिए एक बार compile होता है; op चर runtime पर root से bind होता है।

`SimplifiedPatternMatcher<C>` (`TypedPatternMatcher<C = ()>` इसका alias है) segments की एक सूची है, हर ब्लॉक का एक, प्रत्येक के साथ एक root `OpMask` और closure। `rewrite(node, ctx)` segments को स्कैन करता है, उन्हें छोड़ देता है जिनके mask में नोड का प्रकार नहीं है, और पहला non-`NoMatch` लौटाता है। `a + b` `b` के segments को `a` के बाद जोड़ता है, इसलिए बाएँ operand के नियम जीतते हैं। `with_context::<D>()` एक `TypedPatternMatcher<()>` को `D`-context मैचर में उठाता है (यह `&self` लेता है); हाथ से लिखे closures `add`, `add_rejecting`, `add_wildcard` से जोड़े जाते हैं। `Matcher<C>` trait है (`fn rewrite(&self, &Arc<UOp>, &mut C) -> RewriteResult`); `late/dtype.rs` में `DemoteFloat` इसे सीधे implement करता है।

## Rewrite इंजन {#the-rewrite-engine}

`ir/src/rewrite/engine.rs` Tinygrad के `unified_rewrite` का stack-आधारित port है। हर नोड तीन चरणों से गुज़रता है:

| चरण | क्या होता है |
|-------|--------------|
| 0 — PushChildren | यदि `bpm` मैचर दिया गया है, तो नीचे उतरने से *पहले* इसे इस नोड पर fixpoint तक लागू करें (patterns मूल children देखते हैं)। `Gate(node)` एक प्रतिस्थापन दर्ज करता है और children को छोड़ देता है। फिर children को push करें, फिर इस नोड के लिए एक stage-1 प्रविष्टि। |
| 1 — ApplyPatterns | children को replacement map से resolve करें (यदि कोई तैयार नहीं है तो waitlist)। यदि कोई child बदला, तो नोड को फिर से बनाएँ और पुनर्निर्मित नोड को stage 0 पर वापस भेजें। अन्यथा `pm` लागू करें; `Rewritten` परिणाम stage 0 पर push होता है — पूरी तरह फिर से traverse और फिर से match, यही fixpoint है — एक stage-2 link के साथ। |
| 2 — Link | मूल नोड को उसके प्रतिस्थापन के अंतिम परिणाम से map करें। |

परिणाम `UOp::id` द्वारा memoize होते हैं (`replace`, `bpm_cache`; `Gate` कभी cache नहीं होता)। दो सीमाएँ: `REWRITE_STACK_LIMIT = 500_000` stack प्रविष्टियाँ (`"infinite loop in graph_rewrite (stack too big: ..)"`), और प्रति-नोड `bpm_seen` सेट जो तब panic करता है जब bottom-up fixpoint किसी नोड पर दोबारा आता है। कोई iteration सीमा नहीं है।

| प्रवेश बिंदु | मैचर |
|-------------|----------|
| `graph_rewrite(pm, root, ctx)` | stage 1 पर `pm` — नियम rewrite किए गए children देखते हैं (Tinygrad डिफ़ॉल्ट) |
| `graph_rewrite_bottom_up(bpm, root, ctx)` | stage 0 पर `bpm` — नियम मूल children देखते हैं (Tinygrad `bottom_up=True`); `Gate` का सम्मान होता है |
| `graph_rewrite_with_bpm(pm, bpm, root, ctx)` | दोनों; केवल tests में उपयोग |
| `graph_rewrite_walk(bpm, root, ctx)` | एक पास, प्रतिस्थापन फिर से traverse नहीं होते (Tinygrad `walk=True`) |
| `*_preserve_calls` रूप | वही, लेकिन `CALL`/`FUNCTION` bodies या `PROGRAM` के आंतरिक भाग में प्रवेश किए बिना (Tinygrad `enter_calls=False`) |

`RewriteResult` या तो `NoMatch`, `Rewritten(Arc<UOp>)` या `Gate(Arc<UOp>)` है; `pm` मैचर में `Gate` को `NoMatch` माना जाता है। Kernel cut `Gate` का उपयोग `split_all_stores` को पहले से बने kernel `SINK` में उतरने से रोकने के लिए करता है (`rangeify/kernel.rs`)। यदि कोई नियम वही नोड लौटाता है जो उसे दिया गया था, तो एक `debug_assert` चल जाता है।

एकमात्र non-graph driver `line_rewrite` (`linearize/mod.rs`) है: यह रैखिक instruction सूची पर एक बार चलता है, हर प्रविष्टि को कई में फैलने देता है, और बाद के sources को एक map के माध्यम से प्रतिस्थापित करता है। इसका एकमात्र उपयोगकर्ता `line_rewrite_cleanups` है, यानी gated-`STORE` → `IF`/`STORE`/`ENDIF` विस्तार।

`RUST_LOG=svod_ir::pattern=trace` हर match (`op_key`) को log करता है; यह log नहीं करता कि कौन-सा नियम चला या कौन-से आज़माए गए।

## संयोजन क्रमबद्ध है {#composition-is-ordered}

मैचर `+` द्वारा एक निश्चित क्रम में जोड़े जाते हैं, और क्रम का अर्थ होता है। `symbolic_simple()` `propagate_invalid` से शुरू होता है, क्योंकि अन्यथा `x * 0 → 0` `MUL(0, WHERE(c, x, Invalid))` को उसकी validity सहित मिटा देगा; `with_tier2` term combining से पहले canonicalization और comparison नियमों से पहले ALU folding रखता है, क्योंकि हर समूह अगले के लिए मैच उजागर करता है (देखें [बीजगणितीय सरलीकरण](./algebraic-simplification.md))। नया नियम जोड़ने का अर्थ है यह चुनना कि उस क्रम में वह कहाँ लागू होगा।

## Z3 से rewrites का सत्यापन {#verifying-rewrites-with-z3}

`schedule/src/z3/` (feature `z3`, वैकल्पिक निर्भरता `z3 = "0.21"`, सिस्टम `libz3`; nix flake इसे प्रदान करता है) rewrites पर भरोसा करने के बजाय उन्हें जाँचता है:

- `convert.rs` एक UOp tree को Z3 term में अनुवादित करता है: `CONST` (int, uint, bool; floats और `Invalid` अस्वीकार), `DefineVar` एक सीमित पूर्णांक के रूप में, `RANGE` `0 <= r < end` वाले नए चर के रूप में, `Neg`, पूर्णांक binary ops `Add, Sub, Mul, FloorDiv, FloorMod, CDiv, CMod, Max, Lt, Eq, Ne` (bools पर `And`/`Or`), पूर्णांकों पर `WHERE` और `MulAcc`, और `CAST` एक dtype-सीमित नए चर के रूप में, जो source range फिट होने पर अपने source से बँधा होता है। `alu.rs` `CDiv`/`CMod` को C truncation semantics देता है; floor division उन्हीं पर बना है। बाकी सब `ConversionError` है।
- `verify_equivalence(original, simplified)` दोनों को एक context में बदलता है और `original != simplified` assert करता है: `UNSAT` rewrite को सिद्ध करता है, `SAT` `CounterExample::Found { model, .. }` लौटाता है, timeout पर `Unknown`।

```rust
/// The identity elimination `x + 0 = x` is pointer-identical and Z3-proven.
#[test]
fn z3_verify_identity_add_zero(x in arb_var_uop(DType::Int32)) {
    let zero = UOp::native_const(0i32);
    let expr = x.try_add(&zero).expect("ADD accepts matching dtypes");
    let simplified = rewrite(Matchers::simple(), expr.clone());
    prop_assert!(Arc::ptr_eq(&simplified, &x));
    verify_equivalence(&expr, &simplified).expect("Z3 should verify x + 0 = x");
}
```

क्या कवर है (`schedule/src/test/`): `symbolic_simple` से गुज़रती हाथ से लिखी पंक्तियाँ (`unit/z3/symbolic_patterns.rs`), `symbolic_simple` और `symbolic` से गुज़रते `arb_arithmetic_tree_bounded_up_to` और `arb_known_property_graph` पर proptest oracles (`property/oracles.rs`, प्रत्येक 300–500 cases), और संरचनात्मक symbolic tests का दोहरा रन, जो convert होने पर हर पंक्ति को Z3 से दोबारा जाँचता है (`unit/symbolic/mod.rs`)। केवल `Found` test को विफल करता है; `Unknown` और `ConversionFailed` सहन किए जाते हैं, और एक liveness test सुनिश्चित करता है कि अंकगणितीय core अब भी convert होता है। इसे `cargo test -p svod-schedule --features z3,proptest` से चलाएँ; CI वही features `nix flake check` के माध्यम से चलाता है।

प्रमाण सीमित op उपसमुच्चय के नमूना expressions पर असीमित पूर्णांकों के लिए है — यह index सरलीकारक के लिए एक मज़बूत regression जाल है, हर pattern का सत्यापन नहीं।
