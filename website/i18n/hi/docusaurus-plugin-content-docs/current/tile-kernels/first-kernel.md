---
sidebar_label: आपका पहला कर्नेल
---

# आपका पहला कर्नेल {#your-first-kernel}

tk कर्नेल एक Rust function है जो `Kernel` builder से एक tile program रिकॉर्ड करता है। हर कॉल खुले
block में एक statement जोड़ती है और एक typed handle लौटाती है। Handle अपना tier (`Gmem<T>`,
`Shared<T>`, `Regs<T>`) और element type (`BF16`, `F16`, `F32`, `I32`, `Bool`) साथ रखता है; एक
shared allocation `SmemRef<T>` होता है। Statement रिकॉर्ड होते समय shapes जाँचे जाते हैं। Block
indices, offsets और trip counts जैसे scalars साधारण operators से बने `Sc` expressions हैं।

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
    let mut k = Kernel::new("axpb");
    let x = k.param::<F32>("x", ParamKind::In, rows * cols);
    let y = k.param::<F32>("y", ParamKind::Out, rows * cols);
    k.grid([Sc::from(rows.div_ceil(16)), Sc::from(1), Sc::from(1)]);
    k.warps(4);
    let row0 = k.block(0) * 16;
    let tile = |k: &mut Kernel, p: ParamRef<F32>| {
        let v = k.view(p, 0, [cols, 1], Shape::new(16, cols), [Some(Sc::from(rows)), None]);
        k.at(v, row0.clone(), 0)
    };
    let xv = tile(&mut k, x);
    let v = k.load(xv);
    let two = k.fill::<F32>(Shape::new(16, cols), Const::Float(2.0));
    let one = k.fill::<F32>(Shape::new(16, cols), Const::Float(1.0));
    let v = k.mul_add(v, two, one);
    let yv = tile(&mut k, y);
    k.store(yv, v);
    k.finish()
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
    let mut k = Kernel::new("gemm_nt");
    let a = k.param::<BF16>("a", ParamKind::In, m * kk);
    let b = k.param::<BF16>("b", ParamKind::In, n * kk);
    let c = k.param::<BF16>("c", ParamKind::Out, m * n);
    k.grid([Sc::from(m.div_ceil(bm)), Sc::from(n.div_ceil(bn)), Sc::from(1)]);
    k.warps(4);
    let (row0, col0) = (k.block(0) * bm, k.block(1) * bn);
    let bound = |len: usize, tile: usize| (!len.is_multiple_of(tile)).then(|| Sc::from(len));
    let a_view = k.view(a, 0, [kk, 1], Shape::new(bm, bk), [bound(m, bm), None]);
    let a_view = k.at(a_view, row0.clone(), 0);
    let b_view = k.view(b, 0, [kk, 1], Shape::new(bn, bk), [bound(n, bn), None]);
    let b_view = k.at(b_view, col0.clone(), 0);
    let (sa, sb) = (Shape::new(bm, bk), Shape::new(bn, bk));
    let a_s = k.smem::<BF16>("a_s", stages * sa.elems());
    let b_s = k.smem::<BF16>("b_s", stages * sb.elems());
    let acc = k.zeros::<F32>(Shape::new(bm, bn));
    let [acc] = k.pipeline(
        kk / bk,
        stages,
        [acc],
        |k, step, slot| {
            let koff = step * bk;
            let a_g = k.at(a_view, 0, koff.clone());
            let a_t = k.smem_slot(a_s, slot.clone(), sa);
            k.stage(a_t, a_g, CopyMode::Async);
            let b_g = k.at(b_view, 0, koff);
            let b_t = k.smem_slot(b_s, slot, sb);
            k.stage(b_t, b_g, CopyMode::Async);
        },
        |k, _step, slot, [acc]| {
            let a_t = k.smem_slot(a_s, slot.clone(), sa);
            let b_t = k.smem_slot(b_s, slot, sb);
            [k.mma(acc, a_t, false, b_t, true)]
        },
    );
    let out = k.cast::<F32, BF16>(acc);
    let c_view = k.view(c, 0, [n, 1], Shape::new(bm, bn), [bound(m, bm), bound(n, bn)]);
    let c_view = k.at(c_view, row0, col0);
    k.store(c_view, out);
    k.finish()
}
```

**Padding की जगह bounds.** `bound(len, tile)` सिर्फ़ वहीं `Some(len)` है जहाँ tile grid `len` से आगे
निकलता है, इसलिए कोई भी `m` और `n` बिना copy के चलते हैं। सिर्फ़ `k` का `bk` का गुणज होना ज़रूरी है।

**Pipeline.** `pipeline(extent, stages, init, produce, consume)` `stages` shared slots की ring पर
एक producer और एक consumer घोषित करता है, और `init` (यहाँ accumulator) को consumer के ज़रिए आगे
ले जाता है। लेखक बताता है कि slot में क्या copy करना है और उससे क्या गणना करनी है।
[Schedule template](./layouts-and-lowering#schedule-templates) तय करता है कि copies कितना आगे चलें,
और हर wait और barrier को जगह देता है।

**गुणनफल.** `mma(acc, a, a_t, b, b_t)` f32 accumulators (8-bit integers पर i32, `MmaIn::Acc`) पर `acc + A·B` है,
यहाँ `B` transposed
पढ़ा जाता है। यह कभी किसी instruction का नाम नहीं लेता: lowering target का matrix-core atom चुनती है
और `a`, `b` और `acc` को उसके layouts देती है। नतीजा एक ही बार round होता है, store से पहले के cast
पर।

## इसे device पर चलाना {#running-it-on-the-device}

`launch::graph_launch` हर घोषित parameter के लिए क्रम से एक tensor लेता है और पहला output एक lazy
`Tensor` के रूप में लौटाता है। नतीजा realize होने पर कर्नेल चलता है, किसी भी graph kernel की तरह।
लाइब्रेरी का GEMM एक `GemmCfg` लेता है, जिसके `lowering` को device की मापी गई costs चाहिए।

```rust
use svod_dtype::{DType, default_device::default_device};
use svod_tensor::Tensor;
use svod_tk::atoms::{Core, Device, Target};
use svod_tk::build::BF16;
use svod_tk::kernels::Batch;
use svod_tk::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use svod_tk::launch::graph_launch;

let (m, n, k) = (1000, 512, 256);
let target = Target::for_device(&default_device()).expect("a GPU target");
let device = Device { costs: svod_tk::probe::costs(&target)?.costs, target };
let cfg =
    GemmCfg { tile: [64, 64, 32], stages: 2, warps: [2, 2], unroll: true, core: Core::Warp, crews: 1 };
let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg };
let a = Tensor::empty(&[m * k], DType::BFloat16);
let b = Tensor::empty(&[n * k], DType::BFloat16);
let c = Tensor::empty(&[m * n], DType::BFloat16);
let c = graph_launch(gemm::<BF16>(&spec), &cfg.lowering(&device), &[&a, &b, &c])?;
c.realize()?;
```

मॉडल यह ख़ुद नहीं करते: [`ops::linear`](./op-layer) config चुनता है, output को shape देता है और
कर्नेल लागू न होने पर ग्राफ़ पर लौट जाता है।

## Builder कॉल्स {#builder-calls}

| समूह | कॉल्स |
|---|---|
| घोषणाएँ | `param`, `var` (launch variable जो नाम से bind होता है), `grid`, `warps`, `smem` |
| Scalars | `block(axis)`, `warp()`, `row()` (gather map के भीतर), `load_scalar(param, index)`, `Sc` operators और `min`/`max`/`lt`/`le`/`eq`/`and`/`or` |
| Views | `view`, `gather` (row-mapped view), `at`, `narrow`, `smem_view`, `smem_view_as`, `smem_slot` |
| Tile ops | `fill`, `zeros`, `splat`, `coord`, `unary`, `binary` (row या column vector broadcast होता है), `scale`, `compare` (`CmpOp`), `cast`, `cast_toward_zero`, `where_`, `mul_add` (एक ही rounding), `reduce`, `reduce_f32`, `transpose`, `shift` (axis के साथ एक step आगे, पहले स्थान पर fill), `flip` (हर element axis के साथ mask के bits पार अपने partner से बदलता है: sorting network का एक step), `mma`, `mma_oriented`, `pin` |
| Movement | `stage` (global → shared, `CopyMode::Async` या `Sync`), `load`, `load_as_fill`, `store` |
| Control | `loop_` (carried register tiles), `pipeline`, `if_`, `select_if` (tiles बनाने वाली branches), `where_any` (एक `where_` जिसका चुना हुआ पक्ष केवल वे warps गणना करते हैं जिनके पास predicate का कोई set element है), `each_warp` (एक `Warps` statement), `barrier` |
