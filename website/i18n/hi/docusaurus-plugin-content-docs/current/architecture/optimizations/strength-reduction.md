---
sidebar_label: स्ट्रेंथ रिडक्शन
---

# स्ट्रेंथ रिडक्शन और देर से होने वाले विघटन (Late Decompositions) {#strength-reduction-and-late-decompositions}

देर से होने वाले (late) rewrites operations को सस्ते समतुल्यों से बदलते हैं, और जिन operations की backend में कमी है उन्हें उन operations से बदलते हैं जो उसके पास हैं। ये index lowering के बाद, [post-optimization pipeline](../codegen/linearizer.md) के `19b`–`20` चरणों में चलते हैं, क्योंकि पहले के passes को मूल संरचना की ज़रूरत होती है: `Add(Mul(a, b), c)` को `MulAcc` बनने से पहले term combining को दिखाई देते रहना चाहिए। स्रोत: `schedule/src/optimizer/mod.rs` में `early_decomposition_patterns`, `get_late_rewrite_patterns`, `pm_mod_to_idiv`; नियम `schedule/src/rangeify/patterns.rs` और `schedule/src/symbolic/fast_div.rs` में; `ir/src/decompositions/`। Tinygrad: `codegen/decomp/op.py` (`get_late_rewrite_patterns`, `fast_idiv`)।

## संरचना {#composition}

```text
19b  early_decomposition_patterns(supported)
       symbolic_simple + pm_fold_cast_const + pm_mod_to_and + divmod_decomposition_patterns
       + pm_threefry_decomp              if !supports(Threefry)
       + pm_max_decomposition            if !supports(Max) && supports(Lt)
       + pm_erf_decomposition            if !supports(Erf)

19d  pm_decomp = early
       + get_late_rewrite_patterns(renderer, disable_fast_idiv)
           pm_mod_to_and + pm_half_bf16_cast
           + pm_demorgan                   if supports(Or)
           + pm_mul_to_shl                 if supports(Shl)
           + pm_div_to_shr                 if supports(Shr)
             + fast_division_patterns + pm_mod_to_idiv    if also DISABLE_FAST_IDIV=0
           + pm_neg_from_mul               if supports(Neg)
           + pm_comparison_negations       if supports(Lt) || supports(Eq)
           + pm_fma_decomposition          if supports(MulAcc)
             + pm_shl_add_to_mulacc        if also supports(Shl)
           + pm_fdiv_to_mul                if supports(Fdiv)
       + get_transcendental_patterns(supported, TRANSCENDENTAL >= 2)
       + renderer.decomposition_matcher()  if the device defines one

20   pm_final = pm_commit_weak + pm_cast_weak + pm_decomp (+ extra_matcher) + pm_split_ends
```

`supports(..)` renderer की `RendererOps` तालिका है। हर चरण एक ही fixpoint है, इसलिए नियम एक-दूसरे को आगे बढ़ाते हैं: `pm_mul_to_shl` `R1 * 64` को `R1 << 6` में बदलता है, और फिर `pm_shl_add_to_mulacc` `(R0 << 2) + (R1 << 6)` को `MulAcc(R0, 4, R1 << 6)` में fuse करता है — [worked example](../codegen/worked-example.md) वाला पूर्णांक FMA। `DISABLE_FAST_IDIV` का डिफ़ॉल्ट **1** है: नीचे दिया गया magic-number division opt-in है।

## Floor से truncating division तक {#floor-to-truncating-division}

`divmod_decomposition_patterns` (`ir/src/decompositions/mod.rs`) `FloorDiv`/`FloorMod` को C-शैली के `CDiv`/`CMod` में lower करता है जो हर backend के पास है, और चिह्न सुधार `q - (r != 0 && (a < 0) != (b < 0))` / `r + (correction ? b : 0)` जोड़ता है, जब तक दोनों operands सिद्ध रूप से शून्य के एक ही ओर न हों (`same_truncating_bucket`)। नीचे के सभी power-of-two और magic-number नियम `CDiv`/`CMod` से मेल खाते हैं (या `FloorMod` से, `pm_mod_to_and` के लिए, जो `19b` में भी चलता है ताकि power-of-two modulo lowering से पहले fold हो जाएँ)।

## Power-of-two नियम {#power-of-two-rules}

| नियम | पैटर्न | परिणाम | शर्त |
|------|---------|--------|-------|
| `pm_mod_to_and` | `FloorMod(x, 2^n)` | `x & (2^n - 1)` | पूर्णांक `x` (floor modulo के लिए सटीक, कोई भी चिह्न) |
| `pm_mul_to_shl` | `Mul[x, 2^n]` | `x << n` | पूर्णांक `x` |
| `pm_div_to_shr` | `CDiv(x, 2^n)` | `x >> n` | `vmin(x) >= 0` या unsigned |
| | | `(x + WHERE(x < 0, 2^n - 1, 0)) >> n` | signed `x` जो ऋणात्मक हो सकता है |

यह bias arithmetic shift की −∞ की ओर rounding को truncating division की शून्य की ओर rounding में सुधारता है। LLVM backend पर signed `Shr` `ashr` के रूप में render होता है, इसलिए जब भी `vmin` सिद्ध न किया जा सके, bias आवश्यक है।

## Magic-number division (`fast_division_patterns`, `fast_div.rs`) {#magic-number-division-fast_division_patterns-fast_divrs}

`CDiv(x, d)` के लिए, जहाँ `d` एक धनात्मक non-power-of-two स्थिरांक है और `x` unsigned है या `vmin(x) >= 0`:

1. `magic_unsigned(vmax, d)` — Hacker's Delight: `nc = (vmax + 1) / d * d - 1`, `nbits = 64 - leading_zeros(vmax)`, और सबसे छोटा `s ∈ 0..=2*nbits` जिसके लिए `2^s > nc * (d - 1 - (2^s - 1) % d)`; `M = (2^s + d - 1 - (2^s - 1) % d) / d`। परिणाम `(x * M) >> s` सभी `0 <= x <= vmax` के लिए `x / d` के बराबर है। कॉल `max(vmax, |vmin|)` का उपयोग करती है।
2. यदि `M * vmin` और `M * vmax` dtype में समाते हैं: `(x * M) >> s` उत्पन्न करें।
3. अन्यथा power-of-two गुणनखंड बाहर निकालें: `d = 2^k * d'` `CDiv(x, 2^k)` बन जाता है (पिछले नियम के बाद एक shift) और बिना चौड़ा किए `d'` पर recursion करें।
4. अन्यथा अगले पूर्णांक dtype तक चौड़ा करें (`i8 → i16 → i32 → i64 → u64`, `u8 → u16 → u32 → u64`), जब renderer उसका समर्थन करता हो और गुणनफल वहाँ समाता हो, और वापस cast करें।

इसके बाद `pm_mod_to_idiv` मेल खाते `CMod(x, d)` को `x - d * CDiv(x, d)` के रूप में फिर से लिखता है, ताकि शेषफल भी उसी रास्ते से जाए। `fast_idiv` में एक signed सुधार (`+ (x < 0)`) मौजूद है, लेकिन pattern guard उसे अप्राप्य बना देता है। उदाहरण: `x ∈ [0, 255]`, `d = 7` → `M = 293`, `s = 11`; `(255 * 293) >> 11 = 36 = 255 / 7`।

## Float और FMA {#float-and-fma}

- `pm_fdiv_to_mul`: `Fdiv(x, c)` → `x * (1/c)`, ऐसे float स्थिरांक के लिए जिसमें `c != 0` हो और जिसका व्युत्क्रम परिमित हो।
- `pm_fma_decomposition`: `Add[Mul(a, b), c]` → `MulAcc(a, b, c)` जब तीनों का float dtype एक ही हो। पूर्णांक यहाँ fuse नहीं होते।
- `pm_shl_add_to_mulacc`: `Add[Shl(x, n), c]` → `MulAcc(x, 2^n, c)` — कोई float guard नहीं, इसलिए यह पूर्णांक रास्ता है (`0 <= n < 64`)।
- `pm_neg_from_mul`: `Mul[x, -1]` → `Neg(x)` (एकमात्र स्थान जहाँ `Neg` op बनता है; अन्यत्र `neg()` `MUL(x, -1)` बनाता है), और `Add[x, Neg(y)]` → `Sub(x, y)`।
- `pm_half_bf16_cast`: समान-चौड़ाई वाले float cast (`f16 ↔ bf16`) के लिए कोई एकल LLVM instruction नहीं है, और साधारण `cast(f32).cast(dst)` श्रृंखला को cast नियम वापस fold कर देंगे, इसलिए इसे bits के माध्यम से लिखा जाता है: `f16 → f32 → RNE-round the low 16 bits → bf16`, और `bf16 → (u16 << 16 as f32) → f16`।

## Comparison निषेधन (`pm_comparison_negations`) {#comparison-negations-pm_comparison_negations}

केवल पूर्णांक; स्थिरांक अंकगणित `checked_*` का उपयोग करता है और overflow पर अस्वीकार कर देता है।

| पैटर्न | परिणाम |
|---------|--------|
| `Not(Lt(x, c))` | `Lt(c - 1, x)` |
| `Not(Lt(c, x))` | `Lt(x, c + 1)` |
| `And[Lt(c1, x), Lt(x, c2)]`, `c2 == c1 + 2` | `Eq(x, c1 + 1)` |
| `Lt(Mul(x, -1), c)` | `Lt(-c, x)` |
| `Lt(Mul(x, -1), Mul(y, c))` | `Lt(y * -c, x)` |

`pm_demorgan` देर से लागू होने वाला `And[Not(x), Not(y)]` → `Not(Or(x, y))` है, केवल bool, `Or` पर निर्भर; De Morgan की दोनों दिशाएँ `symbolic()` के `boolean_dsl_patterns` में भी हैं, जिसे `symbolic_simple` (और इसलिए late fixpoint का अपना tier-1 सेट) शामिल नहीं करता।

## Op विघटन {#op-decompositions}

- `pm_max_decomposition`: `Max(a, b)` → `WHERE(a < b, b, a)`।
- `pm_erf_decomposition`: Abramowitz–Stegun 7.1.26, `erf(x) = sign(x) * (1 - t * P(t) * exp(-x²))`, जहाँ `t = 1 / (1 + 0.3275911 |x|)` और `P` Horner बहुपद `1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592` है; अधिकतम त्रुटि लगभग 1.5e-7। `Erf` यहाँ तक UOp बना रहता है क्योंकि `@llvm.erf` एक libm कॉल है जिसे in-process JIT link नहीं करता।
- `pm_threefry_decomp`: Threefry2x32, `u32` अंकगणित में पाँच rounds।
- `get_transcendental_patterns`: f16/f32/f64 के लिए `Exp2`/`Log2`/`Sin` → `xexp2`/`xlog2`/`xsin` (`ir/src/decompositions/transcendentals.rs`), अन्य floats f32 के माध्यम से; `Sqrt` → `xpow(x, 0.5)`; हर एक केवल तब जब renderer में वह op न हो, और `TRANSCENDENTAL=2` होने पर सभी।
- Device का `decompositor()` (Metal: `amd_decomposition_patterns` — native `exp2`/`log2` के ऊपर `Exp`, `Log`, `Cos`, `Tan`, binary `Pow`)।

## Dtype अनुकरण (`19c`) {#dtype-emulation-19c}

शुरुआती और देर के विघटनों के बीच, `pm_dtype_decomp_commit` उन dtypes का अनुकरण करता है जिनका renderer समर्थन नहीं करता: `Int64`/`UInt64` को 32-bit words के जोड़ों के रूप में (`pm_long_decomp`, `Lt` से carries और `CDiv`/`CMod` के लिए 64-चरण shift-subtract divider के साथ), और FP8/`Float16`/`BFloat16` को मूल storage word पर `Float16` या `Float32` गणना के रूप में (`pm_float_decomp`, bit-exact `f2f` रूपांतरण)। चयन हर graph के लिए उस पर एक बार चलकर किया जाता है (`DTypeDecompCtx`); `get_dtype_decomps` compile cache key के लिए वही सूची उपलब्ध कराता है।
