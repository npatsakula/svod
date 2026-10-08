---
sidebar_label: Layouts और Lowering
---

# Layouts और Lowering

`lower::lower(program, &lowering, params, device)` एक tile program को ऐसे Svod program में बदलता है
जिसकी instruction list पहले से क्रम में है। एक `Lowering` में वह सब होता है जो लेखक ने तय नहीं
किया: `Target`, `Schedule`, block tile पर `WarpGrid`, और यह कि shared rows swizzle होंगी या नहीं।
हर कर्नेल config अपना `Lowering` बनाता है (`GemmCfg::lowering`, `FaCfg::lowering`,
`NormCfg::lowering`)।

| क़दम | मॉड्यूल | असर |
|---|---|---|
| 1. Schedule expansion | `schedule::expand` | हर `Pipeline` loops, branches और explicit `Sync` statements बन जाता है |
| 2. Operand materialization | `lower` | Shared या global `mma` operand को एक नए register tile में explicit load मिलता है |
| 3. Barriers | `lower::sync` | Author-level shared memory traffic के आसपास barriers डाले जाते हैं |
| 4. Layout inference | `layouts::infer` | हर register tile को एक layout मिलता है; टकराव पर `Relayout` |
| 5. Emission | `lower/emit.rs` | Statements program order में `UOp::linear_program` में emit होते हैं |

## F2 linear layouts {#f2-linear-layouts}

एक `layout::Layout` GF(2) पर input dimensions के bits से output dimensions के bits तक एक linear map
है (Triton का "Linear Layouts" formulation)। Dimensions `Dim::{Reg, Lane, Warp, Block, Row, Col}`
हैं। Map हर input bit के लिए एक basis vector रखता है। हर vendor fragment, हर XOR swizzle और हर
`ldmatrix` plan ऐसा ही एक map है। यह PTX `mma.m16n8k16` accumulator है, `layout/atoms.rs` से:

```rust
/// PTX `mma.m16n8k16` C/D fragment (16×8, M×N): `row = g + 8·(c/2), col = 2t + c%2`.
pub fn mma_sync_c() -> Layout {
    Layout::from_bases(
        [(Row, 16), (Col, 8)],
        &[(Reg, &[[0, 1], [8, 0]]), (Lane, &[[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]])],
    )
}
```

Algebra में `compose`, `inverse`/`pseudo_inverse`, `product` (एक layout को दूसरे पर tile करना),
`transpose`, `sublayout` और `slice` हैं। `test/unit/layout.rs` के tests हर atom को उसके closed form
से exhaustively जाँचते हैं।

किसी register tile का layout एक `layouts::TileLayout { frag, reps, warps }` है। इसमें
`(Reg, Lane)` पर एक fragment होता है, जो हर warp के sub-tile में `reps` बार दोहराया जाता है, साथ में
warps के sub-tile coordinates।

## Atoms {#atoms}

एक `atoms::MmaAtom` एक matrix-core instruction है, अपने `a`, `b` और `c` operands के layouts के साथ।
`Target::mma(dtype_in, dtype_out)` उसे ढूँढता है। कर्नेल कभी उसका नाम नहीं लेता: inference operand
layouts उस atom से पढ़ता है जो values को consume करता है।

| Target | Atom (bf16/f16 → f32) | Fragment layouts |
|---|---|---|
| CUDA | `mma.sync` m16n8k16 | `mma_sync_a`, `mma_sync_b`, `mma_sync_c` |
| AMD CDNA | MFMA 16×16×16 | `mfma_16x16x16` |
| AMD RDNA3 / RDNA4 | WMMA 16×16×16 | `wmma_gfx11_*` / `wmma_gfx12` |
| Apple | simdgroup 8×8×8 | `simdgroup_8x8` |

आज सिर्फ़ CUDA वाली row के पास kernel tables हैं और वही hardware पर चलती है (देखें [पोर्टेबिलिटी](./portability))।

## Layout inference {#layout-inference}

1. **Atoms बीज बोते हैं।** एक `mma` अपने operands और नतीजे को `WarpGrid` पर tile किए गए atom के
   layouts देता है (`layouts::mma_layouts`)।
2. **Elementwise ops एक करते हैं।** `binary`, `cast`, `where_` के operands और नतीजा, और किसी loop
   की carried values, एक ही layout साझा करते हैं।
3. **बिना बंधन वाले tiles natural layout लेते हैं।** Fixed point पर, जिस tile पर कोई बंधन नहीं, उसे
   `layouts::natural` मिलता है: हर lane अधिकतम 8 elements का एक छोटा row vector रखती है, lanes
   पहले columns फिर rows पर चलती हैं, और warps rows बाँटते हैं। Tiles vectors से पहले seed होते हैं।
4. **Vectors अपने tile के पीछे चलते हैं।** एक `[rows, 1]` या `[1, cols]` vector, जैसे `reduce` का
   output या bias row, उसी row या column layout में माँगा जाता है जो उसका tile सुझाता है। Inference
   उसे उस tile से अलग default नहीं करता।
5. **टकराव relayouts बनते हैं।** जिस value को दो consumers अलग-अलग चाहते हैं, उसके लिए एक
   `TileOp::Relayout` डाला जाता है। `TileLayout::relayout` उसे `Identity`, `RegPermute` (हर lane
   के भीतर), `LaneShuffle` या `ViaSmem` में वर्गीकृत करता है। Emitter फ़िलहाल `LaneShuffle` और
   `ViaSmem` दोनों को shared memory के ज़रिए lower करता है।

## Schedule templates {#schedule-templates}

`schedule::Schedule::Uniform { prefetch, unroll }`: हर warp load भी करता है और compute भी।

| `prefetch` | `pipeline(extent, stages, …)` का expansion |
|---|---|
| `CpAsync`, `stages ≥ 2` | `extent + stages − 1` iterations का एक loop। हर iteration तब तक रुकता है जब तक अधिकतम `stages − 2` copy groups बाक़ी न रहें, फिर barrier करके step `i − (stages − 1)` consume करता है। फिर step `i` की `cp.async` copies को एक iteration पहले ख़ाली हुए slot में भेजता है और उन्हें एक group के रूप में commit करता है (अंत के बाद एक ख़ाली group गिनती को एकसमान रखता है)। |
| `CpAsync`, `stages = 1` | Copy, commit, सबका wait, barrier, consume, barrier। |
| `RegisterStaged` (2 stages) | Step `i` के लिए global → register loads, step `i − 1` consume, register → shared stores, barrier। |

`unroll` loop body को हर ring slot के लिए एक बार copy करता है ताकि slot arithmetic constants में fold
हो जाए। इससे फ़ायदा होता है या नहीं, यह हर config पर मापा जाता है, माना नहीं जाता (देखें
[कर्नेल लाइब्रेरी](./kernel-library#gemm) में GEMM table)।

Async और staged copies template की ज़िम्मेदारी हैं, जो उन्हें fence करता है। बाक़ी `lower::sync`
pass संभालता है: synchronous copy से लिखा गया shared tile किसी दूसरे thread के पढ़ने से पहले fence
होता है, और पिछले fence के बाद पढ़ा गया tile overwrite होने से पहले fence होता है।

## Emission {#emission}

Emitter हर statement के instructions को program order में एक pre-linearized program में सूचीबद्ध
करता है, इसलिए Svod के linearizer का toposort किसी tk3 कर्नेल पर कभी नहीं चलता। View के bounds से
आगे के global accesses gated होते हैं: loads शून्य पढ़ते हैं और stores छोड़ दिए जाते हैं। जब हर row
में chunks की गिनती दो की घात हो, तो shared rows 16-byte chunks में XOR-swizzle होती हैं। 16-bit
tiles के shared से register loads, जहाँ layout अनुमति दे, `ldmatrix.x4` इस्तेमाल करते हैं। CUDA
backend 48 KB की static सीमा से ऊपर की shared memory को एक dynamic array में रखता है। Config
candidates target की opt-in सीमा (`Target::smem_bytes`) के ख़िलाफ़ छाने जाते हैं।

:::note[Emitter के invariants, लेखक के नहीं]
Pure values उस loop level पर सूचीबद्ध होती हैं जिसकी उनके inputs को ज़रूरत है। Register accesses
scalar होते हैं। Shared और global accesses `SHRINK` vector accesses होते हैं। Gated stores एक `If`
के अंदर रहते हैं। Carried values उन्हें परिभाषित करने वाले block के अंत में अपने `phi` में जाती हैं।
ये नियम `lower/emit.rs` में लागू होते हैं, और कर्नेल लेखक को इन्हें कभी नहीं छूना पड़ता।
:::

जो रिकॉर्ड तो होता है पर अभी lower नहीं होता, उसके लिए `lower::Error::Unsupported` लौटाया जाता है:
warp roles, role barriers, `raw` statements, register transposes, और non-CUDA target पर async copies।
`TK3_DUMP_LIST=1` emit की गई list छापता है (देखें [टेस्टिंग और डिबगिंग](./testing-and-debugging))।
