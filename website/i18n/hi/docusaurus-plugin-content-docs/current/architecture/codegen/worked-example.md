---
sidebar_label: उदाहरण सहित विवरण
---

# उदाहरण सहित विवरण: CPU पर एक Row Sum

इस पृष्ठ का हर tree वास्तविक output है। प्रोग्राम:

```rust
let data = Array2::from_shape_fn((8, 64), |(r, c)| (r * 64 + c) as f32);
let x = Tensor::from_ndarray(&data);
let y = x.sum(1)?;
y.realize()?;
```

इसे `SVOD_PER_STAGE_UOPS=1 SVOD_DUMP_STAGE=` (हर post-opt stage) और पहले के passes के लिए JSON subscriber के तहत `RUST_LOG=svod_schedule::rangeify::transforms=debug,svod_schedule::optimizer=debug` के साथ, डिफ़ॉल्ट CPU backend (LLVM, in-process) पर capture किया गया है। Node id आवंटन क्रम हैं और हर run में अलग होंगे; संरचना नहीं बदलेगी।

## Tensor graph

`sum(1)` एक tensor-form `REDUCE` है जो flat 512-element buffer के `PERMUTE` किए गए `RESHAPE` पर चलता है, और `CONTIGUOUS` में लिपटा है क्योंकि परिणाम एक output है:

```text
[16] SINK : Scalar(Void)
└── [15] CONTIGUOUS : Scalar(Float32) shape=[Const(8)]
    └── [14] REDUCE(Add, num_axes=1, ranges=[]) : Scalar(Float32) shape=[Const(8)]
        └── [13] PERMUTE(axes=[1, 0]) : Scalar(Float32) shape=[Const(64), Const(8)]
            └── [12] RESHAPE : Scalar(Float32) shape=[Const(8), Const(64)]
                ├── [11] PARAM(slot=0) : Scalar(Float32) shape=[Const(512)]
                │   └── [0] CONST(Int(512)) : Scalar(WeakInt) shape=[]
                └── [4] STACK(len=2) : Scalar(WeakInt) shape=[Const(2)]
                    ├── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
                    └── [3] CONST(Int(64)) : Scalar(WeakInt) shape=[]
```

`STACK` reshape का shape payload है — shape भी UOp ही होते हैं।

## Rangeify के बाद

Range assignment output को `Weak` range `U0` (8) और reduction को `Reduce` range `U1` (64) देता है; movement ops index `U0 * 64 + U1` में सिमट जाते हैं (tree के लिए [rangeify पृष्ठ](./rangeify.md) देखें)। Kernel cut `STAGE` को `STORE`/`END` में बदलता है, buffers को `PARAM` के रूप में क्रमांकित करता है और range को फिर से क्रमांकित करता है। `apply_pre_optimization` में प्रवेश करने वाली kernel body:

```text
[97] SINK[KERNEL] : Scalar(Void)
└── [96] END : Scalar(Void) shape=[]
    ├── [95] STORE : Scalar(Void) shape=[]
    │   ├── [94] INDEX : Scalar(Float32) shape=[]
    │   │   ├── [84] PARAM(slot=0) : Scalar(Float32) shape=[Const(8)]
    │   │   │   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
    │   │   └── [89] RANGE(R1, Weak) : Scalar(WeakInt) shape=[]
    │   │       └── [2] → (see above)
    │   └── [93] REDUCE(Add, num_axes=0, ranges=[88]) : Scalar(Float32) shape=[]
    │       ├── [92] INDEX : Scalar(Float32) shape=[]
    │       │   ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
    │       │   │   └── [0] CONST(Int(512)) : Scalar(WeakInt) shape=[]
    │       │   └── [91] Add : Scalar(WeakInt) shape=[]
    │       │       ├── [90] Mul : Scalar(WeakInt) shape=[]
    │       │       │   ├── [89] → (see above)
    │       │       │   └── [3] CONST(Int(64)) : Scalar(WeakInt) shape=[]
    │       │       └── [88] RANGE(R0, Reduce) : Scalar(WeakInt) shape=[]
    │       │           └── [3] → (see above)
    │       └── [88] → (see above)
    └── [89] → (see above)
```

Slot 0 output है (`STAGE` का buffer पहले map हुआ था), slot 1 input। पाँचों pre-optimization चरण इस graph को अछूता छोड़ते हैं: कोई movement op नहीं, कोई collapse योग्य reduce नहीं, विभाजित करने के लिए कोई modulo नहीं, मिलाने के लिए कुछ नहीं।

## Optimizer के बाद (`00-initial`)

CPU renderer में locals नहीं हैं, इसलिए `convert_loop_to_global` `R1` को `Weak` ही रहने देता है। `hand_coded_optimizations` tensor cores, image upcasts, matvec path और grouped reductions को छोड़ देता है; `apply_unroll` 64-चौड़ा reduce देखता है (32 से अधिक) और `UNROLL(0, 4)` लागू करता है; kernel पहले से unrolled है, इसलिए `apply_default_upcast` कुछ नहीं करता; 512 elements प्रति-thread 131072 की सीमा से बहुत नीचे हैं, इसलिए कोई `THREAD` नहीं। Kernel का नाम `r_8_16_4` है (reduce; extents 8, 16, 4):

```text
[135] SINK[KERNEL] : Scalar(Void)
└── [131] END : Scalar(Void) shape=[]
    ├── [130] STORE : Scalar(Void) shape=[]
    │   ├── [94] INDEX : Scalar(Float32) shape=[]
    │   │   ├── [84] PARAM(slot=0) : Scalar(Float32) shape=[Const(8)]
    │   │   │   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
    │   │   └── [89] RANGE(R1, Weak) : Scalar(WeakInt) shape=[]
    │   │       └── [2] → (see above)
    │   └── [128] REDUCE(Add, num_axes=0, ranges=[118, 117]) : Scalar(Float32) shape=[]
    │       ├── [122] INDEX : Scalar(Float32) shape=[]
    │       │   ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
    │       │   │   └── [0] CONST(Int(512)) : Scalar(WeakInt) shape=[]
    │       │   └── [121] Add : Scalar(WeakInt) shape=[]
    │       │       ├── [90] Mul : Scalar(WeakInt) shape=[]
    │       │       │   ├── [89] → (see above)
    │       │       │   └── [3] CONST(Int(64)) : Scalar(WeakInt) shape=[]
    │       │       └── [120] Add : Scalar(WeakInt) shape=[]
    │       │           ├── [119] Mul : Scalar(WeakInt) shape=[]
    │       │           │   ├── [118] RANGE(R0, Reduce) : Scalar(WeakInt) shape=[]
    │       │           │   │   └── [115] CONST(Int(16)) : Scalar(WeakInt) shape=[]
    │       │           │   └── [116] CONST(Int(4)) : Scalar(WeakInt) shape=[]
    │       │           └── [117] RANGE(R2, Unroll) : Scalar(WeakInt) shape=[]
    │       │               └── [116] → (see above)
    │       ├── [118] → (see above)
    │       └── [117] → (see above)
    └── [89] → (see above)
```

`apply_opt` ने 64-चौड़े `R0` को `R0 * 4 + R2` में विभाजित किया, जहाँ `R0: Reduce(16)` और `R2: Unroll(4)` हैं, और `pm_flatten_range` ने दोनों को `REDUCE` पर सूचीबद्ध किया। Node गिनती 20।

## `08-post_opt_sym`

केवल `commutative_canonicalization` लागू होता है: index `(R0*4 + R2) + R1*64` बन जाता है (operands का tuplize क्रम)। अब भी 20 nodes।

## `09-pre_expand`

`R2` को `RESHAPE(STACK(0,1,2,3), [4])` से बदल दिया जाता है, हर consumer shaped हो जाता है, और `expand_reduce` lane axis को `num_axes` में ले जाता है:

```text
[157] SINK[KERNEL] : Scalar(Void)
└── [155] END : Scalar(Void) shape=[]
    ├── [154] STORE : Scalar(Void) shape=[]
    │   ├── [94] INDEX : Scalar(Float32) shape=[]
    │   │   ├── [84] PARAM(slot=0) : Scalar(Float32) shape=[Const(8)]
    │   │   │   └── [2] CONST(Int(8)) : Scalar(WeakInt) shape=[]
    │   │   └── [89] RANGE(R1, Weak) : Scalar(WeakInt) shape=[]
    │   │       └── [2] → (see above)
    │   └── [152] RESHAPE : Scalar(Float32) shape=[Const(1)]
    │       ├── [151] REDUCE(Add, num_axes=1, ranges=[118]) : Scalar(Float32) shape=[]
    │       │   ├── [149] INDEX : Scalar(Float32) shape=[Const(4)]
    │       │   │   ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
    │       │   │   │   └── [0] CONST(Int(512)) : Scalar(WeakInt) shape=[]
    │       │   │   └── [148] Add : Scalar(WeakInt) shape=[Const(4)]
    │       │   │       ├── [147] Add : Scalar(WeakInt) shape=[Const(4)]
    │       │   │       │   ├── [119] Mul : Scalar(WeakInt) shape=[]
    │       │   │       │   │   ├── [118] RANGE(R0, Reduce) : Scalar(WeakInt) shape=[]
    │       │   │       │   │   │   └── [115] CONST(Int(16)) : Scalar(WeakInt) shape=[]
    │       │   │       │   │   └── [116] CONST(Int(4)) : Scalar(WeakInt) shape=[]
    │       │   │       │   └── [146] STACK(len=4) : Scalar(WeakInt) shape=[Const(4)]
    │       │   │       │       ├── [29] CONST(Int(0)) : Scalar(WeakInt) shape=[]
    │       │   │       │       ├── [28] CONST(Int(1)) : Scalar(WeakInt) shape=[]
    │       │   │       │       ├── [144] CONST(Int(2)) : Scalar(WeakInt) shape=[]
    │       │   │       │       └── [145] CONST(Int(3)) : Scalar(WeakInt) shape=[]
    │       │   │       └── [90] Mul : Scalar(WeakInt) shape=[]
    │       │   │           ├── [89] → (see above)
    │       │   │           └── [3] CONST(Int(64)) : Scalar(WeakInt) shape=[]
    │       │   └── [118] → (see above)
    │       └── [28] → (see above)
    └── [89] → (see above)
```

`[1]` तक का `RESHAPE` वह placeholder है जो `expand_reduce` reduced lane axis के लिए छोड़ता है; devectorizer इसे हटा देता है।

## `10-pm_reduce`

`reduce_to_acc` accumulator बनाता है। `horizontal_reduce` पहले चारों lanes को fold करता है (`((a0 + a1) + a2) + a3`, हर lane shaped index expression में एक `INDEX`), फिर `R0` पर loop एक register buffer में जमा करता है:

```text
[193] SINK[KERNEL] : Scalar(Void)
└── [192] END : Scalar(Void) shape=[]
    ├── [191] STORE : Scalar(Void) shape=[]
    │   ├── [94] INDEX : Scalar(Float32) shape=[]          ← PARAM(slot=0)[R1]
    │   └── [189] AFTER : Scalar(Float32) shape=[Const(1)]
    │       ├── [165] BUFFER(slot=0, addrspace=Some(Reg)) : Scalar(Float32) shape=[Const(1)]
    │       │   └── [28] CONST(Int(1)) : Scalar(WeakInt) shape=[]
    │       └── [188] END : Scalar(Void) shape=[Const(1)]
    │           ├── [187] STORE : Scalar(Void) shape=[Const(1)]
    │           │   ├── [165] → (see above)
    │           │   └── [186] Add : Scalar(Float32) shape=[Const(1)]
    │           │       ├── [169] AFTER : Scalar(Float32) shape=[Const(1)]
    │           │       │   ├── [165] → (see above)
    │           │       │   ├── [168] STORE : Scalar(Void) shape=[Const(1)]
    │           │       │   │   ├── [167] AFTER : Scalar(Float32) shape=[Const(1)]
    │           │       │   │   │   ├── [165] → (see above)
    │           │       │   │   │   └── [89] → (see above)          ← init inside the R1 loop
    │           │       │   │   └── [166] CONST(Float(0.0)) : Scalar(Float32) shape=[]
    │           │       │   └── [118] RANGE(R0, Reduce) : Scalar(WeakInt) shape=[]
    │           │       │       └── [115] CONST(Int(16)) : Scalar(WeakInt) shape=[]
    │           │       └── [185] Add : Scalar(Float32) shape=[]
    │           │           ├── [182] Add : Scalar(Float32) shape=[]
    │           │           │   ├── [179] Add : Scalar(Float32) shape=[]
    │           │           │   │   ├── [176] INDEX : Scalar(Float32) shape=[]
    │           │           │   │   │   ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
    │           │           │   │   │   └── [175] INDEX : Scalar(WeakInt) shape=[]
    │           │           │   │   │       ├── [148] Add : Scalar(WeakInt) shape=[Const(4)]   ← shaped index
    │           │           │   │   │       └── [29] CONST(Int(0))                             ← lane 0
    │           │           │   │   └── [178] INDEX ... lane 1
    │           │           │   └── [181] INDEX ... lane 2
    │           │           └── [184] INDEX ... lane 3
    │           └── [118] → (see above)
    └── [89] → (see above)
```

(संक्षिप्त: चारों lanes स्थिर lane index को छोड़कर समान हैं।) init store पर `AFTER(acc, [R1])` पर ध्यान दें: `input_ranges` शून्यीकरण को row loop के अंदर रखता है। Stages `11` और `12` कुछ नहीं बदलते — कोई local stage नहीं, कोई GPU range नहीं।

## `13-pm_add_loads` और `14-devectorize`

`pm_expand_broadcast` shaped index के scalar पदों को स्पष्ट करता है (`EXPAND(RESHAPE(R0*4, [1]), [4])`, `R1*64` के लिए भी यही) और `pm_add_loads` register reads और चारों input lanes को `LOAD` में लपेटता है (55 nodes)। फिर `devectorize` सब कुछ scalar बना देता है: shaped index चार scalar `Add` में सिमट जाता है और प्रति-lane `STORE` समूहित हो जाते हैं। `15-early_symbolic` के बाद lanes `LOAD(INDEX(PARAM(1), (R0*4 + R1*64) + k))` के रूप में पढ़ी जाती हैं:

```text
[269] LOAD : Scalar(Float32) shape=[]
└── [268] INDEX : Scalar(Float32) shape=[]
    ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
    └── [253] Add : Scalar(WeakInt) shape=[]
        ├── [119] Mul : Scalar(WeakInt) shape=[]      ← R0 * 4
        └── [90] Mul : Scalar(WeakInt) shape=[]       ← R1 * 64
[305] LOAD : Scalar(Float32) shape=[]
└── [304] INDEX : Scalar(Float32) shape=[]
    ├── [87] → (see above)
    └── [303] Add : Scalar(WeakInt) shape=[]
        ├── [253] → (see above)
        └── [28] CONST(Int(1))
```

`sym` ने index को उस `base + const` रूप में रखा जिसकी अगले stage को ज़रूरत है (49 nodes)।

## `16-memory_coalescing`

चारों loads का base `R0*4 + R1*64` साझा है, offsets 0..3 हैं, और base 4 से विभाज्य है, इसलिए वे एक 4-चौड़ा access बन जाते हैं; lanes `INDEX(load, k)` हैं:

```text
[341] Add : Scalar(Float32) shape=[]
├── [340] Add : Scalar(Float32) shape=[]
│   ├── [339] Add : Scalar(Float32) shape=[]
│   │   ├── [335] INDEX : Scalar(Float32) shape=[]
│   │   │   ├── [334] LOAD : Scalar(Float32) shape=[Const(4)]
│   │   │   │   └── [333] SHRINK : Scalar(Float32) shape=[Const(4)]
│   │   │   │       ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
│   │   │   │       ├── [253] Add : Scalar(WeakInt) shape=[]          ← R0*4 + R1*64
│   │   │   │       └── [116] CONST(Int(4))                            ← width
│   │   │   └── [29] CONST(Int(0))
│   │   └── [336] INDEX ... [334], CONST(1)
│   └── [337] INDEX ... [334], CONST(2)
└── [338] INDEX ... [334], CONST(3)
```

44 nodes। `17-bottom_up_ew_image` और `16-extra_symbolic` यहाँ no-op हैं।

## `17-pm_lower_index_dtype` और `18-final_symbolic`

हर `WeakInt` `Int32` पर स्थिर हो जाता है — range, constants, `PARAM` sizes:

```text
[386] RANGE(R1, Weak) : Scalar(Int32) shape=[]
└── [351] CONST(Int(8)) : Scalar(Int32) shape=[]
[377] RANGE(R0, Reduce) : Scalar(Int32) shape=[]
└── [373] CONST(Int(16)) : Scalar(Int32) shape=[]
[391] Add : Scalar(Int32) shape=[]
├── [390] Mul : Scalar(Int32) shape=[]
│   ├── [377] → (see above)
│   └── [366] CONST(Int(4)) : Scalar(Int32) shape=[]
└── [389] Mul : Scalar(Int32) shape=[]
    ├── [386] → (see above)
    └── [381] CONST(Int(64)) : Scalar(Int32) shape=[]
```

`18-final_symbolic`, `19-cast_float_alu`, `19b` और `19c` कुछ नहीं बदलते: कोई transcendental नहीं, कोई emulated dtype नहीं।

## `19d-late_decompositions` से `20-final_rewrite` तक

Late rewrites `R1 * 64` को `R1 << 6` (`pm_mul_to_shl`), `R0 * 4` को `R0 << 2`, और `(R0 << 2) + (R1 << 6)` को एक integer `MulAcc` (`pm_shl_add_to_mulacc`) में बदलते हैं। Gate movement के पास हिलाने को कुछ नहीं है (कहीं `Invalid` नहीं), और final rewrite के `pm_split_ends` के पास विभाजित करने को कुछ नहीं है (हर `END` पहले से एक ही range बंद करता है)। अंतिम graph, 43 nodes:

```text
[455] SINK[KERNEL] : Scalar(Void)
└── [454] END : Scalar(Void) shape=[]
    ├── [453] STORE : Scalar(Void) shape=[]
    │   ├── [428] INDEX : Scalar(Float32) shape=[]
    │   │   ├── [353] PARAM(slot=0) : Scalar(Float32) shape=[Const(8)]
    │   │   │   └── [351] CONST(Int(8)) : Scalar(Int32) shape=[]
    │   │   └── [386] RANGE(R1, Weak) : Scalar(Int32) shape=[]
    │   │       └── [351] → (see above)
    │   └── [452] LOAD : Scalar(Float32) shape=[]
    │       └── [451] INDEX : Scalar(Float32) shape=[]
    │           ├── [450] AFTER : Scalar(Float32) shape=[Const(1)]
    │           │   ├── [356] BUFFER(slot=0, addrspace=Some(Reg)) : Scalar(Float32) shape=[Const(1)]
    │           │   │   └── [354] CONST(Int(1)) : Scalar(Int32) shape=[]
    │           │   └── [449] END : Scalar(Void) shape=[]
    │           │       ├── [448] STORE : Scalar(Void) shape=[]
    │           │       │   ├── [363] INDEX : Scalar(Float32) shape=[]
    │           │       │   │   ├── [356] → (see above)
    │           │       │   │   └── [361] CONST(Int(0)) : Scalar(Int32) shape=[]
    │           │       │   └── [447] Add : Scalar(Float32) shape=[]
    │           │       │       ├── [418] LOAD : Scalar(Float32) shape=[]
    │           │       │       │   └── [417] INDEX : Scalar(Float32) shape=[]
    │           │       │       │       ├── [415] AFTER : Scalar(Float32) shape=[Const(1)]
    │           │       │       │       │   ├── [356] → (see above)
    │           │       │       │       │   ├── [413] STORE : Scalar(Void) shape=[]
    │           │       │       │       │   │   ├── [412] INDEX : Scalar(Float32) shape=[]
    │           │       │       │       │   │   │   ├── [410] AFTER : Scalar(Float32) shape=[Const(1)]
    │           │       │       │       │   │   │   │   ├── [356] → (see above)
    │           │       │       │       │   │   │   │   └── [386] → (see above)
    │           │       │       │       │   │   │   └── [361] → (see above)
    │           │       │       │       │   │   └── [166] CONST(Float(0.0)) : Scalar(Float32) shape=[]
    │           │       │       │       │   └── [377] RANGE(R0, Reduce) : Scalar(Int32) shape=[]
    │           │       │       │       │       └── [373] CONST(Int(16)) : Scalar(Int32) shape=[]
    │           │       │       │       └── [361] → (see above)
    │           │       │       └── [446] Add : Scalar(Float32) shape=[]
    │           │       │           ├── [445] Add : Scalar(Float32) shape=[]
    │           │       │           │   ├── [444] Add : Scalar(Float32) shape=[]
    │           │       │           │   │   ├── [443] INDEX : Scalar(Float32) shape=[]
    │           │       │           │   │   │   ├── [439] LOAD : Scalar(Float32) shape=[Const(4)]
    │           │       │           │   │   │   │   └── [438] SHRINK : Scalar(Float32) shape=[Const(4)]
    │           │       │           │   │   │   │       ├── [359] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
    │           │       │           │   │   │   │       │   └── [357] CONST(Int(512)) : Scalar(Int32) shape=[]
    │           │       │           │   │   │   │       ├── [437] MulAcc : Scalar(Int32) shape=[]
    │           │       │           │   │   │   │       │   ├── [377] → (see above)
    │           │       │           │   │   │   │       │   ├── [366] CONST(Int(4)) : Scalar(Int32) shape=[]
    │           │       │           │   │   │   │       │   └── [435] Shl : Scalar(Int32) shape=[]
    │           │       │           │   │   │   │       │       ├── [386] → (see above)
    │           │       │           │   │   │   │       │       └── [434] CONST(Int(6)) : Scalar(Int32) shape=[]
    │           │       │           │   │   │   │       └── [366] → (see above)
    │           │       │           │   │   │   └── [361] → (see above)
    │           │       │           │   │   └── [442] INDEX : Scalar(Float32) shape=[]
    │           │       │           │   │       ├── [439] → (see above)
    │           │       │           │   │       └── [354] → (see above)
    │           │       │           │   └── [441] INDEX : Scalar(Float32) shape=[]
    │           │       │           │       ├── [439] → (see above)
    │           │       │           │       └── [399] CONST(Int(2)) : Scalar(Int32) shape=[]
    │           │       │           └── [440] INDEX : Scalar(Float32) shape=[]
    │           │       │               ├── [439] → (see above)
    │           │       │               └── [395] CONST(Int(3)) : Scalar(Int32) shape=[]
    │           │       └── [377] → (see above)
    │           └── [361] → (see above)
    └── [386] → (see above)
```

नीचे से ऊपर पढ़ें: `BUFFER(Reg)` accumulator है; `STORE([412], 0.0)` उसे `AFTER(acc, R1)` के बाद शून्य करता है, यानी प्रति row एक बार; loop body `STORE([363], LOAD(acc) + lanes)` `END(.., R0)` से बंद होती है; अंतिम `LOAD` उस `END` के बाद accumulator पढ़ता है और `R1` पर output में store होता है; बाहरी `END` `R1` को बंद करता है।

## Linearize और render

`linearize` 43 instructions `(run_count, priority, slot, tuplize)` क्रम में emit करता है: दोनों `PARAM` (हर एक से पहले उसका size constant), register `BUFFER` और उसका `INDEX` सबसे पहले (`run_count` 1, priorities −20/−18), फिर `RANGE(R1)`, zero store, `RANGE(R0)`, body, `END(R0)`, output store, `END(R1)`, `SINK`। CPU renderer उस सूची को इसमें बदलता है:

```llvm
define void @r_8_16_4(ptr noalias align 32 %data0, ptr noalias align 32 %data1) #0 {
entry:
  %reg0 = alloca [1 x float]
  %v1 = getelementptr inbounds float, ptr %reg0, i32 0
  br label %loop_entry_1
loop_entry_1:
  br label %loop_latch_1
loop_latch_1:
  %r1 = phi i32 [ 0, %loop_entry_1 ], [ %r1phi, %loop_footer_1 ]
  %r1phi = add i32 %r1, 1
  %r1cmp = icmp ult i32 %r1, 8
  br i1 %r1cmp, label %loop_body_1, label %loop_exit_1
loop_body_1:
  %v3 = getelementptr inbounds float, ptr %reg0, i32 0
  %v4 = shl i32 %r1, 6
  store float 0x0000000000000000, ptr %v3
  br label %loop_entry_0
loop_entry_0:
  br label %loop_latch_0
loop_latch_0:
  %r0 = phi i32 [ 0, %loop_entry_0 ], [ %r0phi, %loop_footer_0 ]
  %r0phi = add i32 %r0, 1
  %r0cmp = icmp ult i32 %r0, 16
  br i1 %r0cmp, label %loop_body_0, label %loop_exit_0
loop_body_0:
  %v7 = getelementptr inbounds float, ptr %reg0, i32 0
  %v8 = load float, ptr %v7
  %v9.mul = mul i32 %r0, 4
  %v9 = add i32 %v9.mul, %v4
  %v10 = getelementptr inbounds float, ptr %data1, i32 %v9
  %v11 = load <4 x float>, ptr %v10
  %v12 = extractelement <4 x float> %v11, i32 0
  %v13 = extractelement <4 x float> %v11, i32 1
  %v14 = extractelement <4 x float> %v11, i32 2
  %v15 = extractelement <4 x float> %v11, i32 3
  %v16 = fadd nsz arcp contract afn float %v12, %v13
  %v17 = fadd nsz arcp contract afn float %v16, %v14
  %v18 = fadd nsz arcp contract afn float %v17, %v15
  %v19 = fadd nsz arcp contract afn float %v8, %v18
  store float %v19, ptr %v1
  br label %loop_footer_0
loop_footer_0:
  br label %loop_latch_0
loop_exit_0:
  %v23 = getelementptr inbounds float, ptr %reg0, i32 0
  %v24 = load float, ptr %v23
  %v25 = getelementptr inbounds float, ptr %data0, i32 %r1
  store float %v24, ptr %v25
  br label %loop_footer_1
loop_footer_1:
  br label %loop_latch_1
loop_exit_1:
  ret void
}
```

चौड़ाई 4 का `SHRINK` `load <4 x float>` बन गया, integer `MulAcc` `mul` + `add` बन गया, register buffer एक `alloca`; बाद में LLVM का अपना optimizer accumulator को register में रखता है। परिणाम `[2016, 6112, 10208, 14304, 18400, 22496, 26592, 30688]` है।

## Dump पढ़ना

| लक्षण | पहले देखने योग्य stages |
|---------|-------------------------|
| गलत मान | `08` (symbolic), `09` (expansion), `10` (accumulator init/identity), `19d` (decompositions) |
| गलत loop गिनती या गायब loop | pre-opt split/simplify ranges, `12` (gpudims), `10` (`END` merging) |
| जहाँ vector loads अपेक्षित थे वहाँ scalar loads | `15`/`16`: index विभाज्य base के साथ `base + const` होना चाहिए, समान buffer, समान validity, कोई gate नहीं |
| अंतिम graph में `WeakInt` | `17-pm_lower_index_dtype` (`SVOD_SPEC` इसे `18` पर पकड़ता है) |
| अंतिम graph में `Invalid` | `19e` gate movement, `20` `pm_remove_invalid` (debug assertion) |
| कोई backend किसी op को अस्वीकार करता है | `19b`/`19d` capability table (`supported_ops`) |

प्रति stage `node_count` सबसे सस्ता संकेत है: छोटे kernel पर जो stage गिनती को दोगुना कर दे, उसी का dump लें।
