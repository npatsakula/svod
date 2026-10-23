---
sidebar_label: Layouts और Lowering
---

# Layouts और Lowering {#layouts-and-lowering}

`lower::lower(program, &lowering, params, device)` एक tile program को ऐसे Svod program में बदलता है
जिसकी instruction list पहले से क्रम में है। एक `Lowering` में वह सब होता है जो लेखक ने तय नहीं
किया: `Device` (target और उसकी costs), `Schedule`, block tile पर `WarpGrid`, और products किस
matrix instruction पर issue होंगे (`core`); shared rows हमेशा XOR-swizzle होती हैं। हर
कर्नेल config अपना `Lowering` बनाता है (`GemmCfg::lowering`, `FaCfg::lowering`,
`NormCfg::lowering`)। `lower::standalone(program, &lowering)` बिना जुड़े device के किसी target के
लिए lower करता है, और `lower::render` वह LLVM text लौटाता है जो backend के compiler को सौंपा जाता।

| क़दम | मॉड्यूल | असर |
|---|---|---|
| 1. Operand materialization | `lower` | Shared या global `mma` operand को एक नए register tile में explicit load मिलता है; warpgroup core पर shared operands अपनी जगह रहते हैं |
| 2. Barriers | `lower::sync` | Shared memory effects fence किए जाते हैं: barriers, और async copy के tile के इस्तेमाल से पहले एक commit और wait। `Pipeline` अभी भी एक statement है, जिसका template अपनी ring ख़ुद fence करता है, इसलिए pass सिर्फ़ वहीं fence करता है जहाँ ring आसपास के code से मिलती है |
| 3. Schedule expansion | `schedule::expand` | हर `Pipeline` loops, branches और explicit `Sync` statements बन जाता है |
| 4. Layout inference | `layouts::infer` | हर register tile को एक layout मिलता है; टकराव पर `Relayout` |
| 5. Shared orientation | `lower` | Registers के ज़रिए भरा गया shared allocation उस axis के साथ रखा जाता है जिस पर उसके readers के fragments चलते हैं, ताकि gather 16-byte loads हो; `cp.async` tiles row-major रहते हैं |
| 6. Emission | `lower/emit/` | Statements program order में `UOp::linear_program` में emit होते हैं |

`lower` `Error::Grid` के साथ fail होता है जब program के warps warp grid के warps न हों, या body में
`Warps` statement होने पर उनका गुणज न हों; और `Error::DivergentBarrier` के साथ जब कोई block barrier
warp पर branch के नीचे हो, जिसे कुछ warps छोड़ देंगे।

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

Algebra में `identity`, `zeros`, `from_bases`, `transpose`, `product` (एक layout को दूसरे पर tile
करना), `sublayout` और `apply` हैं। Tests हर atom को vendor ISA documents के lane formulas
(`test/unit/atoms.rs`) से और tk1 के closed forms (`test/unit/layout.rs`) से जाँचते हैं।

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
| CUDA sm_90 | `wgmma` m64nNk16, चार warps एक साथ | `wgmma_a`, `wgmma_c`; `B` descriptor के ज़रिए, कभी registers में नहीं |
| AMD CDNA | MFMA 16×16×16 | `mfma_16x16x16` |
| AMD RDNA3 / RDNA4 | WMMA 16×16×16 | `wmma_gfx11_input`, `wmma_gfx11_acc` / `wmma_gfx12` |
| Apple | simdgroup 8×8×8 (सिर्फ़ atom; कोई kernels नहीं) | `simdgroup_8x8` |

Kernels वाले हर target में i32 में जाने वाला एक 8-bit integer atom भी है (`Target::geometry_of(Int8)`):
`mma.sync` m16n8k32 s8 और CDNA का MFMA 16×16×32 i8 16-bit fragments के bytes रखते हैं, दोगुनी
गहराई के साथ (`mma_sync_a8`, `mma_sync_b8`, `mfma_16x16x32_i8`); RDNA का `iu8` WMMA 16-bit layouts
को depth 16 पर रखता है। इसके `m` और `n` 16-bit atom वाले ही हैं। Planner 16-bit geometry
(`Target::geometry`) की cost लगाता है, सिवाय 8-bit rows पर किसी problem के।

`atoms::GroupMmaAtom` एक warpgroup instruction है: चार लगातार warps मिलकर 64 × n का product जारी
करते हैं, operands shared memory से descriptors के ज़रिए पढ़े जाते हैं (`B` हमेशा, `A` तब तक जब तक वह
register tile न हो), और accumulator पूरे group पर फैला रहता है, हर warp अपनी 16 rows `mma.sync` के C
layout में रखता है। जिस config का `core` `Core::Group` है, उसके products इसी पर lower होते हैं: grid
की rows warpgroups हैं, shared operands load होने के बजाय अपनी जगह रहते हैं, और emitter हर product को
ring की swizzled rows पर एक `wgmma` chain के रूप में लिखता है (descriptor का swizzle mode lowering का
अपना XOR है, `layouts::Swizzle`)। GEMM और attention दोनों configs sm_90 पर यह core देते हैं।

## Layout inference {#layout-inference}

1. **Atoms बीज बोते हैं।** एक `mma` अपने operands और नतीजे को `WarpGrid` पर tile किए गए atom के
   layouts देता है (`MmaAtom::issue`), और उसे उसी रुख़ (`ir::Orient`) से issue करता है जो lowering की
   `Costs` के तहत program पर सबसे कम relayout cost छोड़े: `Cᵀ = Bᵀ·Aᵀ` वही instruction है जिसके slots
   आपस में बदल दिए गए हों, इसलिए square atom पर accumulator बिना किसी cost के अगले product का operand
   बनकर रखा जा सकता है। जिस product का नतीजा deal होता है (`TileOp::Deal`, किसी list की lane queues)
   वह उसी रुख़ से issue होता है जो एक row को सबसे कम lanes पर फैलाए (`MmaAtom::narrow`): वहीं deal
   registers का नाम बदलना भर है, जैसा op layer उसकी क़ीमत लगाती है। लेखक orientation pin कर सकता है
   (`Kernel::mma_oriented`)।
2. **Elementwise ops एक करते हैं।** `binary`, `cast`, `where_`, `mul_add` के operands और नतीजा, और
   किसी loop की carried values, एक ही layout साझा करते हैं।
3. **बिना बंधन वाले tiles natural layout लेते हैं।** Fixed point पर, जिस tile पर कोई बंधन नहीं, उसे
   `layouts::natural` मिलता है: हर lane अधिकतम 8 elements का एक छोटा row vector रखती है, lanes
   पहले columns फिर rows पर चलती हैं, और warps rows बाँटते हैं।
4. **Vectors अपने tile के पीछे चलते हैं।** एक `[rows, 1]` या `[1, cols]` vector, जैसे `reduce` का
   output या bias row, उसी row या column layout में माँगा जाता है जो उसका tile सुझाता है।
5. **टकराव relayouts बनते हैं।** जिस value को दो consumers अलग-अलग चाहते हैं, उसे एक
   `TileOp::Relayout` मिलता है। `TileLayout::relayout` उसे `Identity`, `RegPermute` (हर lane के
   भीतर, मुफ़्त), `Exchange` (हर register lane-xor moves से तय साथियों से लिया जाता है) या `ViaSmem`
   में वर्गीकृत करता है। `Exchange` तभी तक रखा जाता है जब तक उसका काम shared memory से होकर एक round
   trip से सस्ता हो (`Relayout::work × Costs::lane_move < Costs::smem_trip`)। Emitter `Exchange` को
   target के `atoms::LaneMove` से lower करता है (CUDA पर `shfl.sync`, RDNA पर DPP और `permlanex16`,
   CDNA पर `ds_bpermute`)।

## Schedule templates {#schedule-templates}

`schedule::Schedule::Uniform { fill, unroll, async_products }`: हर warp load भी करता है और compute
भी। `Schedule::PingPong { crew }`: एक register-staged slot पर `crew` warps की दो crews (CDNA GEMM,
`GemmCfg::crews == 2`)।

| Fill | `pipeline(extent, stages, …)` का expansion |
|---|---|
| `Fill::CpAsync`, `stages ≥ 2` | `extent + stages − 1` iterations का एक loop। हर iteration तब तक रुकता है जब तक अधिकतम `stages − 2` copy groups बाक़ी न रहें, फिर barrier करके step `i − (stages − 1)` consume करता है। फिर step `i` की `cp.async` copies को एक iteration पहले ख़ाली हुए slot में भेजता है और उन्हें एक group के रूप में commit करता है (अंत के बाद एक ख़ाली group गिनती को एकसमान रखता है)। |
| `Fill::CpAsync`, `stages ≥ 3`, `async_products` | वही, पर consume की `wgmma` chains अगले fill के दौरान चलती रहती हैं: दो trips पहले consume हुआ slot फिर से भरा जाता है, और हर consume के बाद हर thread अपनी सबसे नई chain को छोड़कर बाक़ी सबका wait करता है (`Sync::WaitProducts`)। |
| `Fill::CpAsync`, `stages = 1` | Copy, wait, consume, और slot फिर से भरने से पहले एक barrier। |
| `Fill::Staged`, 2 stages | एक ही induction variable पर तीन loops: पहला trip step 0 को registers में load करके उसके slot में store करता है; steady trip step `i` को registers में load करता है, fence करता है, step `i − 1` consume करता है, fence करता है, और registers को step `i` के slot में store करता है, उसकी body में कोई branch नहीं (RDNA पर fences loads को products से पहले और commit को उनके बाद रखते हैं); आख़िरी trip सिर्फ़ consume करता है। |
| `Fill::Staged`, 1 stage | एक ही slot पर वही तीन loops: store एक barrier के पीछे तब तक रुकता है जब तक हर warp slot से निपट न जाए, फिर दोबारा barrier, ताकि उसी shared memory में दोगुना गहरा tile समा जाए। |
| `PingPong` | एक trip अपने products पर कटकर load clusters और product clusters में बँटता है, हर एक के बाद block barrier, ताकि एक crew के products बढ़ी issue priority पर registers से चलें जबकि दूसरी crew load कर रही हो। |

`unroll` loop body को हर ring slot के लिए एक बार copy करता है ताकि slot arithmetic constants में fold
हो जाए; tuner दोनों को मापता है। Async और staged copies template की ज़िम्मेदारी हैं, जो उन्हें fence
करता है। बाक़ी `lower::sync` pass संभालता है: synchronous copy से लिखा गया shared tile किसी दूसरे
thread के पढ़ने से पहले fence होता है, और पिछले fence के बाद पढ़ा गया tile overwrite होने से पहले
fence होता है।

## Emission {#emission}

Emitter हर statement के instructions को program order में एक pre-linearized program में सूचीबद्ध
करता है, इसलिए Svod के linearizer का toposort किसी tk कर्नेल पर कभी नहीं चलता। View के bounds से
आगे के global accesses gated होते हैं: loads शून्य पढ़ते हैं और stores छोड़ दिए जाते हैं। जहाँ row
में chunks की गिनती दो की घात हो, वहाँ shared rows 16-byte chunks में XOR-swizzle होती हैं
(`layouts::Swizzle`)। 16-bit tiles के shared से register loads, जहाँ layout अनुमति दे, `ldmatrix.x4`
इस्तेमाल करते हैं, और 8-bit tiles के loads 16×32 bytes पर उसका untransposed रूप; RDNA4 पर global से register loads `global_load_tr_b128` इस्तेमाल करते हैं जहाँ
layout वही हो जो वह छोड़ता है। एक `Warps` statement अपनी body को हर warp पर अकेले, अपनी copies और
barriers के साथ चलाता है। AMD पर f32 `exp2` hardware का `v_exp_f32` है, जो 2⁻¹²⁶ से नीचे के results को
शून्य कर देता है। CUDA पर 48 KB से ऊपर की shared memory एक dynamic array है; candidates
`Target::smem_bytes` के ख़िलाफ़ छाने जाते हैं।

:::note[Emitter के invariants, लेखक के नहीं]
Pure values उस loop level पर सूचीबद्ध होती हैं जिसकी उनके inputs को ज़रूरत है। Register accesses
scalar होते हैं। Shared और global accesses `SHRINK` vector accesses होते हैं। Gated stores एक `If`
के अंदर रहते हैं। Carried values उन्हें परिभाषित करने वाले block के अंत में अपने `phi` में जाती हैं।
:::

`TK_DUMP_LIST=1` emit की गई list छापता है (देखें [टेस्टिंग और डिबगिंग](./testing-and-debugging))।
