---
sidebar_label: बीजगणितीय सरलीकरण
---

# बीजगणितीय सरलीकरण (Algebraic Simplification) {#algebraic-simplification}

सिम्बॉलिक सरलीकारक (symbolic simplifier) `schedule/src/symbolic/patterns.rs` में तीन नेस्टेड मैचरों से बना है। कौन-सा स्तर (tier) कहाँ चलता है:

| मैचर | संरचना | कहाँ चलता है |
|---------|-------------|---------|
| `symbolic_simple()` | `symbolic_simple_base() + dead_loop_patterns()` | add-loads (13), devectorize (14), image पास (17), index lowering (17), शुरुआती decompositions (19b), और अंतिम rewrite के भीतर `pm_decomp` सेट |
| `symbolic()` | `symbolic_simple` + tier-2 समूह | rangeify मेगा-पास, range splitting/merging (`+ pm_fold_cast_const`), `indexing_simplify`, अंतिम symbolic (18) |
| `sym()` | `symbolic` + tier 3 | pre-optimization (`+ pm_fold_cast_const + pm_flatten_range`), post-opt symbolic (08), early symbolic (15), extra symbolic (16, `+ indexing_simplify`) |

`pm_fold_cast_const` (`CAST(CONST) → CONST`) जानबूझकर किसी भी tier के भीतर *नहीं* है; जिन स्थानों को इसकी ज़रूरत है वे इसे स्पष्ट रूप से जोड़ते हैं, ठीक वैसे ही जैसे Tinygrad `symbolic + pm_fold_cast_const` को केवल वहीं जोड़ता है जहाँ `UOp.simplify` ऐसा करता है।

पूर्णांक अंकगणित को फिर से लिखने वाला हर नियम `exact_integer_rewrite` से होकर गुज़रता है, जो एक टाइप्ड no-wrap प्रमाण (`typed_integer_rewrite_is_exact`) है: यदि मूल या प्रतिस्थापन अपने ठोस dtype में overflow कर सकता है, तो वह rewrite को अस्वीकार कर देता है। *value-sensitive* चिह्नित समूह अतिरिक्त रूप से `value_sensitive` में लिपटे होते हैं, जो उन्हें तब तक निष्क्रिय रखता है जब तक subtree के लिए `weak_float_values_are_committed` सत्य न हो। सीमाएँ (bounds) `VminVmaxProperty` (हमेशा उपलब्ध) और `SoundVminVmaxProperty` (उन ops के लिए `None` जिनकी सीमाएँ भरोसेमंद नहीं: loads, `Pow`, `Fdiv`) से आती हैं; दोनों प्रति नोड कैश होती हैं।

## संरचना {#composition}

```text
symbolic_simple_base()         tier 1
  propagate_invalid                     must be first (before x*0 → 0)
  fold_invalid_load_store
  constant_folding_dsl_patterns         value-sensitive
  vconst_folding_patterns               value-sensitive
  bool_arithmetic_patterns
  identity_and_zero_patterns            value-sensitive
  self_folding_dsl_patterns
  zero_folding_dsl_patterns
  division_dsl_patterns                 value-sensitive
  cast_dsl_patterns
  uint_pack_dsl_patterns
  div_mod_recombine_dsl_patterns
  power_dsl_patterns                    value-sensitive
  boolean_dsl_simple_patterns
  dce_dsl_simple_patterns               value-sensitive
symbolic_simple() = base + dead_loop_patterns

symbolic() = symbolic_simple + tier 2 (with_tier2)
  commutative_canonicalization
  boolean_dsl_patterns
  term_combining_dsl_patterns           value-sensitive
  dce_dsl_patterns
  where_alu_combining_patterns
  vmin_vmax_collapse_patterns           value-sensitive
  minmax_dsl_patterns                   value-sensitive
  alu_folding_dsl_patterns              value-sensitive
  comparison_dsl_patterns               value-sensitive
  range_based_mod_div_patterns
  advanced_division_dsl_patterns
  range_based_cast_patterns
  long_to_int_narrowing_patterns
  after_simplification_patterns
  where_bound_patterns                  value-sensitive

sym() = symbolic + tier 3
  pm_simplify_valid                     (symbolic/valid_simplification.rs)
  alu_vectorize_reorder_patterns
  ne_zero_fold_patterns                 value-sensitive
  cast_where_dsl_patterns
  store_load_folding_patterns
  reduce_sym_patterns                   value-sensitive
  sym_phase3_patterns
```

क्रम मायने रखता है: term combining से पहले canonicalization, comparison और range नियमों से पहले ALU folding, क्योंकि हर समूह अगले समूह के लिए मैच उजागर करता है। Tinygrad के reciprocal distribution नियम (`uop/symbolic.py` `sym`) जानबूझकर अनुपस्थित हैं: सभी छह IEEE के अनुसार अयथार्थ (inexact) हैं।

**संकेतन।** `OP[a, b]` क्रमविनिमेय (commutative) है, `OP(a, b)` क्रमबद्ध; `@zero`/`@one`/`c` स्थिरांक हैं; दोहराया गया नाम एक ही नोड है (`Arc::ptr_eq`)। `//` का अर्थ `FloorDiv` है, `%` का अर्थ `FloorMod`; truncating `CDiv`/`CMod` केवल late decompositions के बाद दिखाई देते हैं।

## Tier 1 {#tier-1}

### Invalid का प्रसार (`propagate_invalid`) {#invalid-propagation-propagate_invalid}

`Invalid` है `UOp::invalid_marker()`, एक `ConstValue::Invalid` स्थिरांक; `is_invalid_marker` पूरी तरह `Invalid` वाले `VCONST` या `STACK` और उसके चारों ओर के movement wrappers को भी पहचानता है। वैधता (validity) `WHERE(cond, x, Invalid)` के रूप में रखी जाती है, और ये नियम उस आकार को बरकरार रखते हैं जबकि अंकगणित उसके इर्द-गिर्द घूमता है:

| पैटर्न | परिणाम |
|---------|--------|
| `WHERE(Invalid, _, _)` | `Invalid` |
| `WHERE(WHERE(c, x, Inv), a, b)` | `WHERE(c, WHERE(x, a, b), Inv)` |
| `WHERE(c, Inv, x)` | `WHERE(!c, x, Inv)` (`WHERE(c, Inv, Inv)` → `Inv`) |
| `WHERE(c1, WHERE(c2, x, d), d)` | `WHERE(c1 & c2, x, d)` |
| `WHERE(a, WHERE(c, x, Inv), y)`, `y` `Invalid` नहीं | `WHERE(!a \| c, WHERE(a, x, y), Inv)` — और false शाखा के लिए इसका दर्पण रूप |
| `unary(Inv)`, `CAST(Inv)`, `BITCAST(Inv)` | `Inv` |
| `unary(WHERE(c, x, Inv))`, `CAST`, `BITCAST` | `WHERE(c, unary(x), Inv)` |
| `op(WHERE(c, x, Inv), y)`, `op(y, WHERE(c, x, Inv))` **हर** binary op के लिए, comparisons सहित | `WHERE(c, op(x, y), Inv)` |
| `op(Inv, y)`, `op(y, Inv)` 13 non-comparison binary ops के लिए | `Inv` |

पहले क्यों: `MUL(0, WHERE(c, x, Inv))` को `WHERE(c, 0, Inv)` बनना चाहिए, न कि `0`।

### मृत loads और stores (`fold_invalid_load_store`) {#dead-loads-and-stores-fold_invalid_load_store}

`LOAD(INDEX(buf, Invalid, ..))` (`CAST` के पीछे भी) → load का `alt` यदि हो, अन्यथा उचित आकार का शून्य; gate के बिना `STORE(INDEX(buf, Invalid, ..), v)` → `NOOP`।

### स्थिरांक folding {#constant-folding}

Unary: `Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc` (`Neg` यहाँ कोई op नहीं है — `neg()` `MUL(x, -1)` बनाता है)। Binary: `Add, Mul, Sub, FloorMod, Max, Pow, FloorDiv, Fdiv, And, Or, Xor, Shl, Shr`, साथ में `Bool` देने वाले छह comparisons। Ternary: `Where`, `MulAcc`। परिणाम dtype के storage format से होकर commit होते हैं (`Int32` wrap होता है), सिवाय weak dtypes के, जो बिना truncate किया मान रखते हैं। `vconst_folding_patterns` यही काम lane-दर-lane `VCONST ⊕ VCONST` और `CONST`/`VCONST` broadcast मिश्रण के लिए करता है (11 binary ops + comparisons, 6 unary), weak lanes को छोड़ते हुए।

### Bool अंकगणित {#bool-arithmetic}

जब दोनों `Bool` हों: `Mul[x, y]` → `x & y`, `Add[x, y]` → `x | y`, `Max(x, y)` → `x | y`।

### Identity और शून्य {#identity-and-zero}

| पैटर्न | परिणाम | शर्त |
|---------|--------|-------|
| `Add[x, 0]` | `x` | float नहीं, या शून्य `-0.0` है (`x = -0.0` के लिए `x + 0.0` identity नहीं है) |
| `Sub(x, 0)` | `x` | float नहीं, या शून्य `+0.0` है |
| `Mul[x, 1]`, `Or[x, 0]`, `Xor[x, 0]`, `FloorDiv(x, 1)`, `Fdiv(x, 1)` | `x` | |
| `FloorMod(x, 1)` | `0` | |
| `Floor/Ceil/Trunc/Round(x)` | `x` | पूर्णांक `x` |
| `Mul[x, 0]` | `0` | float नहीं (`NaN * 0`, `Inf * 0` का परिणाम `NaN` है) |
| `And[_, 0]` | `0` | |

### Self और शून्य folding {#self-and-zero-folding}

`FloorDiv(x, x)` → `1`; `FloorDiv(x, -1)` → `MUL(x, -1)`; `FloorMod(FloorMod(x, y), y)` → `FloorMod(x, y)`; `And(x, x)`, `Or(x, x)`, `Max(x, x)` → `x`; `FloorMod(x, x)` → `0`; `Lt(x, x)` → `false` non-floats के लिए, और floats के लिए तब जब sound bounds सिद्ध करें कि `x` `NaN` नहीं है; `Ne(x, x)` → `false` ints और bools के लिए।

### भाग (Division) {#division}

`Fdiv(0.0, 0.0)` और `Fdiv(MUL[_, 0.0], 0.0)` → `NaN` (पहले सूचीबद्ध ताकि ये अगले नियम से पहले लागू हों); `Fdiv(x, x)` → `1.0` केवल तब जब `x` सिद्ध रूप से परिमित और शून्येतर हो; `FloorDiv(Mul(x, y), y)` → `x`। float के लिए `(x*y)/y → x` नहीं है।

### Casts {#casts}

`CAST(x, dt)` → `x` जब dtype पहले से मेल खाता हो; `CAST(CAST(x, a), b)` → `x` जब `x: b` और `can_safe_cast(b, a)` (`a` में `b` का हर मान समा जाता है: समान signedness और कम-से-कम उतना चौड़ा, unsigned→signed के लिए एक अतिरिक्त bit चाहिए, float↔int कभी नहीं); `CAST(CAST(x, a), b)` → `CAST(x, b)` जब `a` `x` को संकीर्ण न करे। `uint_pack_dsl_patterns` Threefry द्वारा बनाई गई `(hi.cast(u64) << 32) | lo.cast(u64)` packing को रद्द करता है, ताकि PRNG 32-bit ALU में ही रहे।

### Div-mod पुनर्संयोजन {#div-mod-recombination}

हर `Add` पर एक नियम: `fold_add_divmod_recombine`, Tinygrad का port। यह `Add` श्रृंखला को सपाट करता है, एक पद `(base % div) * mul` और एक साथी `q * (div * mul)` ढूँढता है जिसका `q` किसी ऐसी चीज़ का भागफल है जो `div` मॉड्यूलो `base` के सर्वांगसम है (`quotient_base`: `q == b // div`, संभवतः merged `(x//c + a)//div` और shifted स्थिरांकों के साथ), और इस जोड़ी को `b * mul` से बदल देता है; `q == (b // div) % d` होने पर यह इसे चौड़े `(b % (div*d)) * mul` में fold करता है। यह `x%n + (x//n)*n → x` परिवार और उसके scaled, offset और तीन-पद वाले रूप हैं, जो अलग-अलग नियमों के बजाय श्रृंखला के माध्यम से मिलते हैं।

### Power, booleans, DCE {#power-booleans-dce}

`Pow(x, 0)` → `1`, `Pow(x, 1)` → `x`, `Pow(1, x)` → `1` (केवल scalars; कोई अन्य घातांक नहीं बदला जाता — reciprocal/sqrt रूप IEEE rounding बदल देते हैं)। `Not(Not(x))` → `x`, `Xor(x, x)` → `0`, `true | _` → `true`, `false & _` → `false`, `true & x` → `x`, `false | x` → `x` (केवल bool स्थिरांक)। सिद्ध रूप से स्थिर शर्त (sound bounds) वाला `WHERE` शाखा चुन लेता है; `WHERE(_, t, t)` → `t`; `WHERE(x, true, false)` → `x`; `WHERE(x, false, true)` → `!x`; `WHERE(a, WHERE(b, c, d), d)` → `WHERE(a & b, c, d)`। `dead_loop_patterns`: `vmax < 0` वाला `RANGE` → `CONST(0)`, `vmin == vmax` वाला `RANGE(CONST)` → वही स्थिरांक। यहाँ कोई `END`/`REDUCE` खाली-range fold नहीं है; उन्हें `reduce_to_acc` संभालता है।

## Tier 2 {#tier-2}

### क्रमविनिमेय canonicalization {#commutative-canonicalization}

`Add, Mul, Max, And, Or, Xor` (और नाममात्र के लिए `Eq`/`Ne`, जो कभी लागू नहीं होते क्योंकि उनका परिणाम `Bool` है) के लिए, जिनका **परिणाम** dtype `WeakInt` है: operands की अदला-बदली तब होती है जब `tinygrad_tuplize_cmp(b, a) == Less` हो — यह संरचनात्मक `(op, arg, dtype, *src)` key क्रम है जिसे linearizer भी उपयोग करता है। इसके बाद commutativity तक समान index expressions hash-cons होकर एक ही नोड बन जाते हैं, जिस पर पुनर्संयोजन और expander निर्भर हैं। अन्य dtypes लिखा गया क्रम बनाए रखते हैं।

### पद संयोजन (स्थिरांकों के साथ `Add`/`Mul`) {#term-combining-addmul-with-constants}

| पैटर्न | परिणाम |
|---------|--------|
| `Add(x, x)` | `x * 2` |
| `Add(Mul[x, c1], Mul[x, c2])` | `x * (c1 + c2)` |
| `Add[x, Mul[x, c]]` | `x * (c + 1)` |
| `Add[Add[y, Mul[x, c0]], Mul[x, c1]]` | `y + x * (c0 + c1)` |
| `Add[Add[y, x], Mul[x, c]]`, `Add[Add[y, Mul[x, c]], x]` | `y + x * (c + 1)` |
| `Add[Add[y, x], x]` | `y + x * 2` |
| `Mul[-1, Add[x, c]]` | `-x + (-c)` |
| `Mul[c, Add[x, k]]`, `x: WeakInt` | `c*x + c*k` |

### Boolean (`boolean_dsl_patterns`) {#boolean-boolean_dsl_patterns}

`Or[x, Not(x)]` → `true`, `And[x, Not(x)]` → `false` (केवल bool); दोनों दिशाओं में De Morgan, `And[Not(x), Not(y)]` → `!(x | y)` और `Or[Not(x), Not(y)]` → `!(x & y)`।

### WHERE {#where}

`dce_dsl_patterns`: `WHERE(Not(c), t, f)` → `WHERE(c, f, t)`, जब तक `f` में `Invalid` न हो (scalar या `STACK` lane) — अदला-बदली marker को true शाखा में ले जाएगी जहाँ gate नियम उसे देख नहीं सकते। `where_alu_combining_patterns`: `op(WHERE(c, a, b), WHERE(c, d, e))` → `WHERE(c, op(a, d), op(b, e))` `Add, Mul, Sub, Max, And, Or, Xor` के लिए, जब दोनों true शाखाएँ या दोनों false शाखाएँ स्थिरांक हों, और साहचर्य (associative) रूप `Add(Add(y, WHERE(c, ..)), WHERE(c, ..))`। `where_bound_patterns`: `WHERE(Lt(x, c), t, f)` → `t` जब `x.vmax < c.vmin`, `f` जब `x.vmin >= c.vmax`।

### सीमाओं का संकुचन और min/max {#bounds-collapse-and-minmax}

`vmin_vmax_collapse_patterns`: एक `Mul`, `FloorDiv`, `FloorMod`, comparison, `PARAM` या `SPECIAL` जिसकी sound bounds एक ही मान हों, वह स्थिरांक बन जाता है (floats बाहर; `Add`/`Sub`/`Max` जानबूझकर बाहर ताकि trip-1 loop carry fold न हो जाए)। `minmax_dsl_patterns`: `Max(x, y)` → `x` जब `x.vmin >= y.vmax` (floats के लिए सख्ती से बड़ा, ताकि शून्य का चिह्न बना रहे), `y` के लिए सममित। कोई `Min` op नहीं है: `Tensor::minimum` एक `WHERE` है।

### ALU श्रृंखला folding (`alu_folding_dsl_patterns`) {#alu-chain-folding-alu_folding_dsl_patterns}

साहचर्य folding `(x ⊕ c1) ⊕ c2` → `x ⊕ (c1 ⊕ c2)` `Add`, `Mul`, `And`, `Or`, `Xor`, `Max` के लिए; स्थिरांक को आगे धकेलना `(x + c) + y` → `(x + y) + c` और `(x * c) * y` → `(x * y) * c` जब `y` स्थिरांक न हो; `(x - c1) + c2`, `(x + c1) - c2` को `x + k` या `x - |k|` में सामान्यीकृत किया जाता है; `(x - c1) - c2` → `x - (c1 + c2)`; `Sub(a, Sub(b, x))` → `x + (a - b)`। Svod में `Sub` एक प्रथम-श्रेणी op है (Tinygrad `a - b` को `a + b*-1` लिखता है)।

### Comparisons (`comparison_dsl_patterns`) {#comparisons-comparison_dsl_patterns}

सभी छह comparisons के लिए: non-floats पर `x op x` fold होता है (`Lt/Gt/Ne` → `false`, `Le/Ge/Eq` → `true`); स्थिरांक operands fold होते हैं; अन्यथा `ComparisonAnalyzer::analyze` (`ir/src/uop/comparison_analysis.rs`) sound bounds से `true` या `false` सिद्ध करता है — दोनों केवल non-weak dtypes के लिए। फिर: `Lt(Add[c0, x], c1)` → `Lt(x, c1 - c0)`; `Lt(Mul[x, -1], Mul[y, -1])` → `Lt(y, x)`; `Lt(FloorDiv(x, d), c)` → `Lt(x, c * d)` `d > 0` के लिए no-wrap जाँच के तहत (floor division के लिए सटीक, `c` का कोई भी चिह्न); `WeakInt` के लिए: `Lt(Mul[c0, x], c1)` → `±x < ceil(c1 / |c0|)` और GCD fold `lt_folding` (`x = d*q + r`, `r ∈ [0, d)`, `d | c` ⇒ `x < c ⇔ q < c/d`)।

### Ranges और भाग {#ranges-and-division}

`range_based_mod_div_patterns` और `advanced_division_dsl_patterns` index बीजगणित हैं; वे [index arithmetic](./index-arithmetic.md) पृष्ठ पर हैं। `range_based_cast_patterns` strong integer `x` के लिए `CAST(CAST(x, a), b)` को संकुचित करता है जिसकी सीमाएँ `a` में समाती हैं। `long_to_int_narrowing_patterns` एक `Int64` binary op को, जिसके operands और परिणाम `i32` में समाते हों, `Int32` op और वापसी cast के रूप में फिर से लिखता है, और signed-int cast को `WeakInt + c` पर वितरित करता है।

### AFTER {#after}

`after_simplification_patterns`: जो deps side effects नहीं हैं (`RANGE`, `STORE`, `END`, `CALL`, `BARRIER`, `CUSTOM`, `FUNCTION`), उन्हें उनके अपने sources से बदला जाता है और दोहराव हटाया जाता है; `NOOP` deps और `END(NOOP)` श्रृंखलाएँ हटा दी जाती हैं; `AFTER(x, [])` → `x`।

## Tier 3 (`sym`) {#tier-3-sym}

- **`pm_simplify_valid`** (`valid_simplification.rs`): `Bool` validity clauses की `And` श्रृंखला को clause-दर-clause सरल किया जाता है (`simplify_valid`), और `x: WeakInt` वाला `WHERE(cond, x, Invalid)` `x` को उन सीमाओं के तहत फिर से लिखता है जो `cond` से निहित हैं (`uop_given_valid`): `parse_valid` हर clause को `expr < c` / `expr >= c` के रूप में पढ़ता है, एक सीमित चर प्रतिस्थापित करता है और फिर से सरल करता है।
- **`alu_vectorize_reorder_patterns`**: `op(STACK(x, x, ..), STACK(y, y, ..))` → `STACK(op(x, y), ..)` 13 अंकगणितीय/bitwise ops और छह comparisons के लिए, जब दोनों operands एक ही नोड के broadcasts हों और lane संख्या समान तथा 1 से अधिक हो।
- **`ne_zero_fold_patterns`**: `Ne(x, 0)` → `x.cast(bool)`।
- **`cast_where_dsl_patterns`**: `CAST(WHERE(s, a, b))` → `WHERE(s, CAST(a), CAST(b))`।
- **`store_load_folding_patterns`**: `STORE(_, Invalid)` → `NOOP`; `STORE(INDEX, WHERE(c, v, Invalid))` → gated index `INDEX(buf, WHERE(c, idx, Invalid))` जो `v` store करता है; `STORE(idx, LOAD(idx))` → `NOOP`; `STORE(INDEX, WHERE(g, alt, LOAD(same INDEX)))` → `alt` का gated store।
- **`reduce_sym_patterns`**: `REDUCE(x * c, Add)` → `REDUCE(x, Add) * c` और `reduce_mul_chain_sym` (range-स्वतंत्र गुणनखंड `Add`/`Max` reduce से बाहर निकाले जाते हैं; `Max` के लिए केवल अऋणात्मक), केवल पूर्णांक।
- **`sym_phase3_patterns`**: `-1 * (x + y)` → `-x + -y`; `WeakInt` के लिए `(x + y) * c` → `x*c + y*c`; एकल-source `GROUP` का unwrap; `NOOP`/`STACK`/`SINK` को `SINK`/`GROUP` में सपाट करना; `END(NOOP)` → `NOOP`।

## उदाहरण: सरलीकरण की श्रृंखला {#worked-cascade}

`x: Int32` के साथ `(x + 0) * 1 + (3 + 4)`:

```text
Add(x, 0)        → x          identity_and_zero
Mul(x, 1)        → x          identity_and_zero
Add(3, 4)        → 7          constant_folding
Add(x, 7)                     stays: no rule
```

इंजन parents से पहले children को फिर से लिखता है और पुनर्निर्मित parent को फिर से मैच करता है, इसलिए तीनों चरण एक ही `graph_rewrite` में होते हैं। Identity वाले चरण वही हैं जिन्हें [Z3 oracles](./pattern-system.md#verifying-rewrites-with-z3) सिद्ध करते हैं।
