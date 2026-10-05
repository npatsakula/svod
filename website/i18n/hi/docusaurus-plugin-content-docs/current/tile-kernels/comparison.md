---
sidebar_label: tk बनाम HipKittens बनाम CuTile
---

# एक tile कर्नेल लिखने के तीन तरीक़े

tile abstraction `tk` की ईजाद नहीं है। यह tile-based कर्नेल systems के एक छोटे-से परिवार का हिस्सा है,
और `tk` के design को समझने का सबसे अच्छा तरीक़ा है इसे अपने दो सबसे क़रीबी रिश्तेदारों के बग़ल में रखकर देखना:

- **[HipKittens](https://github.com/HazyResearch/HipKittens)** — AMD matrix cores के लिए HazyResearch की C++ tile library,
  जिससे `tk` के abstractions का सीधा वंश आता है।
- **[CuTile](https://github.com/NVIDIA/cutile-rs)** (cutile-rs) — NVIDIA GPUs पर tile कर्नेल के लिए NVIDIA Research का Rust system।

तीनों [Tiling क्या है](./tiling) वाला core विचार साझा करते हैं: fragment-sized tiles को registers में खींचो,
उन पर compute करो, और वापस लिख दो। फ़र्क़ इस बात में है कि *hardware mapping को कौन नियंत्रित करता है* — और
यह फ़र्क़ एक spectrum की तरह फैला हुआ है।

---

## यह spectrum: explicit control से लेकर कंपाइलर द्वारा प्रबंधित abstraction तक

तीनों systems एक ही axis पर बैठते हैं। एक छोर पर registers, shared memory, और instruction scheduling — सब
आप ख़ुद manage करते हैं; दूसरे छोर पर आप बस tile-level code लिखते हैं और एक downstream compiler तय करता है कि
यह threads, shared memory, और matrix instructions पर कैसे map होगा। HipKittens explicit छोर पर बैठता है और
CuTile उस छोर पर जहाँ कंपाइलर ही सब प्रबंधित करता है। `tk` बीच से थोड़ा बाएँ बैठता है: यह आपको HipKittens की
तरह explicit register और shared tiles देता है, पर एक standalone backend होने के बजाय Svod के single UOp IR
में lower होता है।

---

## आमने-सामने

| Axis | **tk** | **HipKittens** | **CuTile** |
|------|--------|----------------|------------|
| **Authoring surface** | Rust *builder API* (`Kernel`/`Group` UOps mint करते हैं) | C++ *templates* | Rust *macro DSL* — `#[cutile::module]` में plain Rust लिखो, macro AST capture करता है |
| **IR target** | Svod का **एक UOp IR** — पूरे compiler के समान | कोई नहीं (templates → clang amdgcn) | एक *अलग* MLIR `cuda_tile` dialect, Tile IR bytecode में serialized |
| **Lowering** | Svod render → LLVM → AMD binary, या → PTX (`ptxas` से cubin में assemble, वरना driver का JIT), या → MSL | clang | bytecode → external `tileiras` assembler → cubin (पहले launch पर JIT) |
| **Memory model** | **explicit** register *और* shared tiles | explicit register *और* shared tiles | **एक** tile type (register-resident); shared-mem staging implicit है, compiler ख़ुद चुनता है |
| **Matrix-core API** | explicit `WMMA` op + role-based fragments | typed tiles → `__builtin_amdgcn_mfma_*` | एक single functional `mma()` intrinsic |
| **Compute/memory overlap** | एक `sched::pipeline` marker + एक codegen pass | हर कर्नेल में हाथ से लिखा (raw scheduling intrinsics) | `tileiras` के हवाले |
| **Headline differentiator** | एक IR ⇒ hand कर्नेल और autotuned कर्नेल बराबर के peers हैं | "hardware up से बना" | launch boundary के आर-पार memory safety |
| **Target** | AMD CDNA3 / RDNA3 / RDNA4, NVIDIA `sm_80+`, Apple7+ | AMD CDNA / RDNA | केवल NVIDIA `sm_80+` |

इसके ऊपर हर `tk` कर्नेल अपना arch set ख़ुद declare करता है: flash attention और square `matmul` हर family पर
चलते हैं, `gemm_nt` और norms RDNA और CUDA पर, single-query attention AMD और CUDA पर, k-means और k-NN AMD पर —
पूरी matrix [कर्नेल लाइब्रेरी](./kernel-library) में है।

---

## code कैसा दिखता है

तीनों authoring surfaces का अंदाज़ सचमुच अलग है। HipKittens और CuTile वाले snippets बस समझाने के लिए हैं —
ये हर model का *shape* दिखाते हैं, कोई exact API नहीं; `tk` वाला असली API है।

**HipKittens** — C++ templates; आप tiles को नाम देते हैं और multiply सीधे call करते हैं:

```cpp
using namespace kittens;
rt_bf<64, 32>      a, b;     // register tiles of bf16
rt_fl<64, 32, col> acc;      // fp32 accumulator, col layout (MFMA output)

load(a, a_global, {row, k});
load(b, b_global, {k, col});
mma_ABt(acc, a, b, acc);     // acc += a · bᵀ  → __builtin_amdgcn_mfma_*
```

**CuTile** — एक module के अंदर आम Rust लिखो जिसे macro capture कर लेता है; tiles immutable हैं, और
compiler आपके लिए shared memory stage कर देता है:

```rust
#[cutile::module]
mod kernels {
    use cutile::core::*;
    pub fn gemm(a: &Tensor<f32, A>, b: &Tensor<f32, B>, c: &mut Tensor<f32, C>) {
        let (i, j) = (tile_block_id_x(), tile_block_id_y());
        let mut acc = Tile::<f32, ACC>::zeros();
        for k in 0..a.dim(1) / BK {
            acc = mma(a.partition(AK).load([i, k]),
                      b.partition(BK).load([k, j]),
                      acc);            // one functional intrinsic
        }
        c.partition(CC).store([i, j], acc);
    }
}
```

**tk** — एक Rust builder जो IR mint करता है; आप fragments role से माँगते हैं और `Group` ops emit करते हैं
(`tk/src/kernels/gemm.rs` में `gemm_core` की single-buffered strip, संक्षिप्त रूप में):

```rust
let (outs, ins) = ker.bind_abi(&[GlSpec::new(&[1, 1, n, n], DType::Float32)], &[a_spec, b_spec]);
let g = ker.group_2d(cfg.warps_m, cfg.warps_n);
let a_smem = ker.shared_sw((cfg.block_m, k_step), bf16, TileLayout::Row);   // swizzled LDS strip
let acc = g.zero(ker.acc((reg_m, reg_n), TileLayout::Col));                 // role, not a hardcoded shape

let lp = ker.loop_static(trips);
let a_f = g.fill_local_nobar(a_smem, a_gl, &a_idx, 2);                      // global → LDS, collaborative
let a_sub = g.load(ker.operand((reg_m, k_step), bf16, TileLayout::Row),    // LDS → registers
                   a_f.subtile((reg_m, k_step), (warp_row, 0)), MoveIdx::default());
let acc = g.mma_ab(acc, &a_sub, &bb);                                      // → WMMA UOp
let ended = lp.close();
let _ = g.store(c_gl, acc.after(&ended), MoveIdx::block((0, 0, mrow, nidx), 2));
ker.finish(1)                                                               // SINK { opts_to_apply: Some(vec![]) }
```

CuTile का example एक आम program जैसा पढ़ने में आता है; `tk` का example एक graph बनाने जैसा। यही असली
सौदा है: CuTile का macro आपके *syntax* को capture करके उसे दोबारा parse करता है, जबकि `tk` एक library है
जिसके method calls *ही* IR construction होते हैं।

---

## असल conceptual फ़र्क़

बाक़ी सबसे ज़्यादा दो distinctions मायने रखती हैं।

**Shared memory का मालिक कौन है।** CuTile के पास ठीक *एक* tile concept है — register tile — और यह
shared-memory staging को जान-बूझकर छिपा देता है; data, LDS, caches, और matrix cores से होकर कैसे बहेगा,
यह इसका `tileiras` assembler तय करता है। `tk` और HipKittens register और shared tiles, *दोनों* expose करते
हैं और staging आपसे explicitly करवाते हैं। CuTile register/shared distinction से एक level *ऊपर* बैठता है;
`tk` ठीक उसी *पर*। control की कीमत भी यही है और ताक़त भी: manage करने को ज़्यादा है, पर वे
[overlap और swizzle के फ़ैसले](./where-flops-hide) जो performance जिताते हैं, अब आपके हाथ में हैं।

**IR कहाँ रहता है।** यही `tk` की असली ख़ास चाल है। HipKittens एक standalone C++ framework है — यह कर्नेल
produce करता है, बस इतना ही। CuTile एक *अलग* MLIR dialect में lower होता है जिसे सिर्फ़ इसका अपना toolchain
consume करता है। जबकि `tk` **उसी UOp IR में lower होता है जो बाक़ी Svod पहले से बोलता है।** एक `tk` कर्नेल
किसी दूसरे compiler को सौंपा गया कोई artifact नहीं है — यह तो एक ही IR में एक subgraph है, हर autotuned
कर्नेल के बग़ल में।

:::tip[GPU विशेषज्ञों के लिए]
IR-target का फ़र्क़ toolchain level पर ठोस रूप में दिखता है। `tk` अपने `SINK` को `svod-codegen` से होकर
LLVM IR और फिर या तो एक AMD binary में, या PTX में (जिसे `ptxas` assemble करता है, वरना driver JIT कर देता
है), या Metal पर MSL में render करता है — वही path जो graph कर्नेल लेते हैं। CuTile इसके बजाय
अपने tile dialect को bytecode में serialize करता है, जिसे एक *external* `tileiras` assembler cubin में बदलता
है, और पहले launch पर JIT-compile होता है; HipKittens तो clang से compile हुए C++ templates ही है। तो `tk`
के लिए "एक IR" का सचमुच मतलब है एक ही render-and-compile pipeline, जबकि बाक़ी एक अलग compiler में bridge करते हैं।
:::

---

## यह क्यों ज़रूरी है

यही वह चीज़ है जो Svod को दो ऐसी चीज़ें एक साथ देने देती है, जो आम तौर पर एक साथ हो ही नहीं सकतीं
(mutually exclusive): compiler को schedule ढूँढने देना, और schedule ख़ुद लिखना — वह भी बिना किसी दूसरे
compiler के।

एक BEAM-autotuned matmul और एक हाथ से लिखा Flash Attention — दोनों एक ही DAG में बस `SINK` UOps हैं। दोनों
एक ही renderer से render होते हैं, एक ही runtime पर run होते हैं, और एक ही debugger से print होते हैं।
इन्हें अलग करने वाली इकलौती चीज़ है `opts_to_apply` marker, जिसका घर
[IR में authoring](./lowering) है: वही IR एक optimizer-driven और एक hand-driven, दोनों कर्नेल साथ लिए चलता है।

HipKittens साबित करता है कि hardware-up जाकर आप vendor libraries की बराबरी कर सकते हैं। CuTile साबित करता है
कि GPU कर्नेल को safe और high-level बनाया जा सकता है। `tk` जिस बात पर टिका है वह कहीं संकरी पर Svod के लिए
कहीं ज़्यादा काम की है: hardware-up tile model लो, और उसके इर्द-गिर्द एक नया backend खड़ा करने के बजाय,
*उसी IR को बोलो जो compiler के पास पहले से है*। यही पूरी वजह है कि `tk` छोटा है — और यही वजह कि एक हाथ से
लिखा कर्नेल किसी escape hatch जैसा नहीं, बल्कि एक first-class citizen जैसा लगता है।
