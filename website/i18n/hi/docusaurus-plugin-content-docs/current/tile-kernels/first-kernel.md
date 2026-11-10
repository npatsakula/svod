---
sidebar_label: आपका पहला कर्नेल
---

# आपका पहला कर्नेल {#your-first-kernel}

tk कर्नेल एक Rust function है जो `Kernel` builder से एक tile program रिकॉर्ड करता है:
`Kernel::record(name, |k| …)` कर्नेल खोलता है और उसका `Program` लौटाता है, जो crate के बाहर
केवल पढ़ा जा सकता है (`name()`, `params()`, `body()`)। हर कॉल खुले block में एक statement जोड़ती
है और एक typed handle लौटाती है। Handle अपना tier (`Gmem<T>`, `Shared<T>`, `Regs<T>`), element
type (`BF16`, `F16`, `I8`, `F32`, `I32`, `Bool`) और अपने कर्नेल का brand साथ रखता है — एक
lifetime जो किसी और कर्नेल के पास नहीं — इसलिए किसी और कर्नेल में इस्तेमाल किया गया handle, या
अपने stage के बाद रखा गया pipeline slot, compile error है; एक shared allocation `SmemRef<T>`
होता है। Statement रिकॉर्ड होते समय shapes जाँचे जाते हैं। Block indices, offsets और trip counts
जैसे scalars कर्नेल के scalars और integers से साधारण operators द्वारा बने `Sc` expressions हैं,
वे भी branded। `Sc` opaque है: जब कोई कॉल उसे consume करती है, builder उसे program में intern
करता है, और कोई scalar id कर्नेल से बाहर नहीं जाता।

## एक न्यूनतम कर्नेल {#a-minimal-kernel}

`[rows, 64]` f32 matrix पर `y = 2·x + 1`, हर block में 16 rows। View की row bound की वजह से आख़िरी
block की `rows` से आगे की rows शून्य पढ़ी जाती हैं और उनके writes छोड़ दिए जाते हैं, इसलिए `rows` का
16 का गुणज होना ज़रूरी नहीं।

```rust
use svod_tk::build::*;
use svod_tk::interp::run;
use svod_tk::ir::*;

fn axpb(rows: usize) -> Program {
    let cols = 64;
    Kernel::record("axpb", |k| {
        let x = k.input::<F32>("x", rows * cols);
        let y = k.output::<F32>("y", rows * cols);
        k.grid([Sc::from(rows.div_ceil(16)), Sc::from(1), Sc::from(1)]);
        k.warps(4);
        let (row0, shape) = (k.block(0) * 16, Shape::new(16, cols));
        let xv = k.view(x, 0, [cols, 1], shape, [Some(Sc::from(rows)), None]);
        let xv = k.at(xv, row0.clone(), 0);
        let v = k.load(xv);
        let two = k.fill::<F32>(shape, Const::Float(2.0));
        let one = k.fill::<F32>(shape, Const::Float(1.0));
        let v = k.mul_add(v, two, one);
        let yv = k.view(y, 0, [cols, 1], shape, [Some(Sc::from(rows)), None]);
        let yv = k.at(yv, row0, 0);
        k.store(yv, v);
    })
}

// On the host, no GPU: one Vec<f64> per parameter in, every parameter out.
let x: Vec<f64> = (0..40 * 64).map(f64::from).collect();
let out = run(&axpb(40), vec![x, vec![0.0; 40 * 64]], &[])?;
assert_eq!(out[1][64 * 39 + 1], 2.0 * (64.0 * 39.0 + 1.0) + 1.0);
```

`view(param, offset, [row_stride, col_stride], shape, bounds)` किसी parameter में एक window है।
`at(view, rows, cols)` उसे खिसकाता है और bounds को नए origin के सापेक्ष रखता है।

## एक GEMM {#a-gemm}

bf16 `a [m, k]` और `b [n, k]` पर `c = a·bᵀ`, f32 में accumulate होते हुए। यह `kernels/gemm.rs` का एक
छोटा रूप है, जो इसके ऊपर epilogues (bias, activation, gated weights, scale, residual, f32 output),
grouped tile order, split reductions, warpgroup core और two-crew ping-pong जोड़ता है।

```rust
fn gemm_nt(m: usize, n: usize, kk: usize, [bm, bn, bk]: [usize; 3], stages: usize) -> Program {
    assert!(kk.is_multiple_of(bk), "k is a multiple of bk");
    Kernel::record("gemm_nt", |k| {
        let a = k.input::<BF16>("a", m * kk);
        let b = k.input::<BF16>("b", n * kk);
        let c = k.output::<BF16>("c", m * n);
        k.grid([Sc::from(m.div_ceil(bm)), Sc::from(n.div_ceil(bn)), Sc::from(1)]);
        k.warps(4);
        let (row0, col0) = (k.block(0) * bm, k.block(1) * bn);
        let bound = |len: usize, tile: usize| (!len.is_multiple_of(tile)).then(|| Sc::from(len));
        let a_view = k.view(a, 0, [kk, 1], Shape::new(bm, bk), [bound(m, bm), None]);
        let a_view = k.at(a_view, row0.clone(), 0);
        let b_view = k.view(b, 0, [kk, 1], Shape::new(bn, bk), [bound(n, bn), None]);
        let b_view = k.at(b_view, col0.clone(), 0);
        let (sa, sb) = (Shape::new(bm, bk), Shape::new(bn, bk));
        let a_s = k.ring::<BF16>("a_s", sa.elems(), stages);
        let b_s = k.ring::<BF16>("b_s", sb.elems(), stages);
        let acc = k.zeros::<F32>(Shape::new(bm, bn));
        let [acc] = k.pipeline(
            kk / bk,
            a_s,
            [acc],
            |k, step, slot| {
                let koff = step * bk;
                let a_g = k.at(a_view, 0, koff.clone());
                let a_t = k.slot(a_s, &slot, sa);
                k.stage(a_t, a_g, CopyMode::Async);
                let b_g = k.at(b_view, 0, koff);
                let b_t = k.slot(b_s, &slot, sb);
                k.stage(b_t, b_g, CopyMode::Async);
            },
            |k, _step, slot, [acc]| {
                let a_t = k.slot(a_s, &slot, sa);
                let b_t = k.slot(b_s, &slot, sb);
                [k.mma(acc, a_t, false, b_t, true)]
            },
        );
        let out = k.cast::<F32, BF16>(*acc);
        let c_view = k.view(c, 0, [n, 1], Shape::new(bm, bn), [bound(m, bm), bound(n, bn)]);
        let c_view = k.at(c_view, row0, col0);
        k.store(c_view, out);
    })
}
```

**Padding की जगह bounds.** `bound(len, tile)` सिर्फ़ वहीं `Some(len)` है जहाँ tile grid `len` से आगे
निकलता है, इसलिए कोई भी `m` और `n` बिना copy के चलते हैं। सिर्फ़ `k` का `bk` का गुणज होना ज़रूरी है।

**Pipeline.** `ring(name, slot, schedule)` `slot` elements के उतने shared slots allocate करता
है जितने `schedule` चलाता है (`schedule.stages()`)। Slots की गिनती, जैसे यहाँ, एक rolled ring
है; `Schedule::ring(stages, unroll)` `unroll` होने पर loop को हर slot पर एक बार copy करता है, और
`Schedule::ping_pong(kslice)` एक slot पर दो crews चलाता है।
`pipeline(extent, ring, init, produce, consume)` अपनी ring के schedule के अनुसार चलता है: वह ring के slots पर एक producer और
एक consumer घोषित करता है, और `init` (यहाँ accumulator) को consumer के ज़रिए आगे ले जाता है।
उसके stages जिस किसी और ring तक पहुँचते हैं, उसमें उतने ही slots होने चाहिए: अपने allocation से
आगे जाने वाला shared view lowering को fail करता है (`lower::Error::SharedPast`)। लेखक बताता है
कि slot में क्या copy करना है और उससे क्या गणना करनी है।
[Schedule template](./layouts-and-lowering#schedule-templates) तय करता है कि copies कितना आगे चलें,
और हर wait और barrier को जगह देता है। Slot उसी stage का है जो उसे रखता है: `slot` producer को
लिखने योग्य tile देता है और consumer को केवल पढ़ने योग्य।

**गुणनफल.** `mma(acc, a, a_t, b, b_t)` f32 accumulators (8-bit integers पर i32, `MmaIn::Acc`)
पर `acc + A·B` है, यहाँ `B` transposed पढ़ा जाता है। यह कभी किसी instruction का नाम नहीं लेता:
lowering target का matrix-core atom चुनती है और `a`, `b` और `acc` को उसके layouts देती है। एक
accumulator (`Acc`) `Copy` नहीं है: product उसे consume करके अगला लौटाता है, और loop body अपने
carried accumulators लौटाती है, इसलिए किसी product द्वारा consume हो चुके accumulator को आगे देना
compile error है। यह एक lint है, ownership की गारंटी नहीं: `*acc` वह register tile है जो वह
रखता है (epilogue के लिए), और कोई भी register tile `into` से `Acc` बन जाता है, इसलिए product से
पहले रखी गई copy अगर फिर accumulator बने तो वह एक stale value ले जाती है जिसे न compiler मना
करता है न lowering। नतीजा एक ही बार round होता है, store से पहले के cast पर।

## इसे device पर चलाना {#running-it-on-the-device}

`launch::graph_launch` हर घोषित parameter के लिए क्रम से एक tensor लेता है और पहला output एक lazy
`Tensor` के रूप में लौटाता है। नतीजा realize होने पर कर्नेल चलता है, किसी भी graph kernel की तरह।
लाइब्रेरी का GEMM अपना spec और config साथ में `Admitted<GemmSpec>` के रूप में लेता है।
`GemmSpec::admitted(&planner, cfg)` उसे तब बनाता है जब planner का admission pass हो (config
target पर spec की problem के लिए feasible, उसकी shape पर claimed); केवल `unsafe`
`Admitted::assume(spec, cfg)` बिना admission के config लेता है, admission की कोई जाँच किए बिना।
Admitted spec केवल पढ़ा जा सकता है, और उसके config के `lowering` को device की मापी गई costs
चाहिए।

```rust
use svod_dtype::{DType, default_device::default_device};
use svod_tensor::Tensor;
use svod_tk::atoms::{Device, Target};
use svod_tk::build::BF16;
use svod_tk::kernels::Batch;
use svod_tk::kernels::gemm::{Epilogue, GemmSpec, gemm};
use svod_tk::launch::graph_launch;
use svod_tk::ops::config::Planner;

let (m, n, k) = (1000, 512, 256);
let target = Target::for_device(&default_device()).expect("a GPU target");
let device = Device { costs: svod_tk::probe::costs(&target)?.costs, target };
let planner = Planner::new(device).expect("a target with an SM count");
let epilogue = Epilogue::DEFAULT;
let ranked = planner.gemm_ranked(1, m, n, k, &epilogue);
let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue };
let spec = spec.admitted(&planner, ranked[0]).expect("the ranking admitted it");
let a = Tensor::empty(&[m * k], DType::BFloat16);
let b = Tensor::empty(&[n * k], DType::BFloat16);
let c = Tensor::empty(&[m * n], DType::BFloat16);
let c = graph_launch(gemm::<BF16>(&spec), &spec.cfg().lowering(&planner.device), &[&a, &b, &c])?;
c.realize()?;
```

मॉडल यह ख़ुद नहीं करते: [`ops::linear`](./op-layer) config चुनता है, output को shape देता है और
कर्नेल लागू न होने पर ग्राफ़ पर लौट जाता है।

## Builder कॉल्स {#builder-calls}

| समूह | कॉल्स |
|---|---|
| घोषणाएँ | `input`, `output`, `inout`, `var` (launch variable जो नाम से bind होता है, किसी non-empty range पर), `grid`, `warps` (एक गिनती, warps का एक column, या `WarpGrid` जिस पर register tiles फैलती हैं; `lone_warps` से पहले set होता है), `lone_warps` (grid से आगे के warps, जो केवल `each_warp` के भीतर चलते हैं), `smem` |
| Scalars | `block(axis)`, `warp()`, `row()` (gather map के भीतर), `load_scalar(param, index)` (एक integer parameter, `I8` या `I32`), `Sc` operators और `min`/`max`/`lt`/`le`/`eq`/`and`/`or` |
| Views | `view`, `gather` (row-mapped view), `periodic` (एक table दोहराते columns), `at`, `narrow`, `smem_view`, `smem_slot`, `ring`, `ring_runs` (अपनी pipeline चलाने वाले हर warp के लिए schedule के slots का एक run), `slot`, `slot_part` (stage के slot का एक हिस्सा), `ring_scratch` (`unsafe`: ring को scratch की तरह, जहाँ कोई stage उसे नहीं छूता) |
| Tile ops | `fill`, `zeros`, `splat`, `coord`, `unary`, `binary` (row या column vector broadcast होता है), `scale`, `compare` (`CmpOp`), `cast`, `cast_toward_zero`, `where_`, `mul_add` (एक ही rounding), `reduce`, `reduce_f32`, `transpose`, `shift` (axis के साथ एक step आगे, पहले स्थान पर fill), `flip` (हर element axis के साथ mask के bits पार अपने partner से बदलता है: sorting network का एक step, rotary half का partner), `deal`/`deal_col` (हर lane के पास row के जो columns हैं उनमें से एक, और उनके indices), `mma`, `mma_oriented`, `pin` |
| Movement | `stage` (global → shared, `CopyMode::Async` या `Sync`), `load`, `load_as_fill`, `store` |
| Control | `loop_` (body के through carried accumulators; `loop_vec` plain tiles carry करता है), `pipeline` (`pipeline_vec` plain tiles carry करता है), `if_`, `select_if` (tiles बनाने वाली branches; `select_if_vec` एक `Vec` पर), `where_any` (एक `where_` जिसका चुना हुआ पक्ष केवल वे warps गणना करते हैं जिनके पास predicate का कोई set element है), `if_any` (एक branch जिसे केवल वही warps चलाते हैं, बाद में कोई select नहीं), `each_warp` (एक `Warps` statement), `barrier` |

कोई tile केवल वहीं पढ़ा जाता है जहाँ उसकी definition read को dominate करती है: loop body, branch
या `each_warp` से रखा गया और उनके बाहर पढ़ा गया tile lowering को fail करता है
(`lower::Error::Escaped`)। ऐसी branch के नीचे block barrier, जिसका predicate warp के साथ बदलता है
(जैसे `where_any` और `if_any` का), और जिसे कुछ warps छोड़ देते हैं, वह भी lowering को fail करता है
(`lower::Error::DivergentBarrier`)।
