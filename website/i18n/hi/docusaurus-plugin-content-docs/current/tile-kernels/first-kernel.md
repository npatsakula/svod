---
sidebar_label: आपका पहला कर्नेल
---

# आपका पहला कर्नेल

tk3 कर्नेल एक Rust function है जो `Kernel` builder से एक tile program रिकॉर्ड करता है। हर कॉल खुले
block में एक statement जोड़ती है और एक typed handle लौटाती है। Handle अपना tier (`Gmem<T>`,
`Shared<T>`, `Regs<T>`) और element type (`BF16`, `F16`, `F32`, `I32`, `Bool`) साथ रखता है।
Statement रिकॉर्ड होते समय shapes जाँचे जाते हैं। Block indices, offsets और trip counts जैसे
scalars साधारण operators से बने `Sc` expressions हैं।

## एक न्यूनतम कर्नेल {#a-minimal-kernel}

`[rows, 64]` f32 matrix पर `y = 2·x + 1`, हर block में 16 rows। View की row bound की वजह से आख़िरी
block की `rows` से आगे की rows शून्य पढ़ी जाती हैं और उनके writes छोड़ दिए जाते हैं, इसलिए `rows` का
16 का गुणज होना ज़रूरी नहीं।

```rust
use svod_tk3::build::*;
use svod_tk3::interp::run;
use svod_tk3::ir::*;

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
    let v = k.binary(v, two, BinaryOp::Mul);
    let v = k.binary(v, one, BinaryOp::Add);
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

## GEMM, क़दम दर क़दम {#the-gemm-step-by-step}

`kernels/gemm.rs` `c = act(a·bᵀ + bias) + residual` (या gated weight से `act(gate)·up`) की गणना
लगभग 180 lines में करता है, जिनमें इसके spec और config types भी शामिल हैं। नीचे के अंश हूबहू हैं।

**Parameters और grid.** Buffers flat हैं और batch capacity के हिसाब से आकार लेते हैं। Bound batch
variable grid z बन जाता है।

```rust
let a = k.param::<T>("a", ParamKind::In, cap * m * kk);
let b = k.param::<T>("b", ParamKind::In, halves * n * kk);
let bias = epi.bias.then(|| k.param::<T>("bias", ParamKind::In, halves * n));
let residual = epi.residual.then(|| k.param::<T>("residual", ParamKind::In, cap * m * n));
let c = k.param::<T>("c", ParamKind::Out, cap * m * n);
let (gm, gn) = (m.div_ceil(bm), n.div_ceil(bn));
let (gz, bb) = batch.axis(&mut k);
k.grid([Sc::from(gm), Sc::from(gn), gz]);
k.warps(cfg.warps[0] * cfg.warps[1]);
```

**Padding की जगह bounds वाले views.** `bound(len, tile)` सिर्फ़ वहीं `Some(len)` है जहाँ tile grid
`len` से आगे निकलता है। कोई भी `m` और `n` बिना copy के चलते हैं। सिर्फ़ `k` का `bk` का गुणज होना
ज़रूरी है।

```rust
let (bx, by) = tile_order(&mut k, gm, gn, cfg.group_m);
let (row0, col0) = (bx * bm, by * bn);
let (m_bound, n_bound) = (bound(m, bm), bound(n, bn));
let a_view = k.view(a, batch_offset(&bb, m * kk), [kk, 1], Shape::new(bm, bk), [m_bound.clone(), None]);
let a_view = k.at(a_view, row0.clone(), 0);
```

`tile_order` `group_m` tile rows के समूहों में चलता है ताकि resident blocks L2 में B साझा करें।

**Pipeline.** `pipeline(extent, stages, init, produce, consume)` `stages` shared slots की ring पर
एक producer और एक consumer घोषित करता है। लेखक बताता है कि slot में क्या copy करना है और उससे क्या
गणना करनी है। [Schedule template](./layouts-and-lowering#schedule-templates) तय करता है कि copies
कितना आगे चलें और हर wait और barrier कहाँ रखा जाए।

```rust
k.pipeline(
    trips,
    stages,
    init,
    |k, step, slot| {
        let koff = step * bk;
        for (src, alloc, shape) in
            std::iter::once((a, a_s, sa)).chain(bs.into_iter().zip(b_s).map(|(b, s)| (b, s, sb)))
        {
            let g = k.at(src, 0, koff.clone());
            let t = k.smem_slot::<T>(alloc, slot.clone(), shape);
            k.stage(t, g, CopyMode::Async);
        }
    },
    |k, _step, slot, accs| {
        let a_t = k.smem_slot::<T>(a_s, slot.clone(), sa);
        let mut i = 0;
        accs.map(|acc| {
            let b_t = k.smem_slot::<T>(b_s[i], slot.clone(), sb);
            i += 1;
            k.mma(acc, a_t, false, b_t, true)
        })
    },
)
```

`mma(acc, a, a_t, b, b_t)` f32 accumulators पर `acc + A·B` है। यह कभी किसी instruction का नाम नहीं
लेता: lowering target का matrix-core atom चुनती है और `a`, `b` और `acc` को उसके layouts देती है।

**Epilogue** f32 accumulator पर चलता है, और नतीजा store पर एक ही बार round होता है। Bias एक
`[1, bn]` row vector है जो tile पर broadcast होता है, और residual एक पूरा tile है।

```rust
if let Some(residual) = residual {
    let r = tile(&mut k, residual);
    let r = load_f32(&mut k, r);
    out = k.binary(out, r, BinaryOp::Add);
}
let out = k.cast::<F32, T>(out);
let c_view = tile(&mut k, c);
k.store(c_view, out);
k.finish()
```

## इसे device पर चलाना {#running-it-on-the-device}

`launch::graph_launch` हर घोषित parameter के लिए क्रम से एक tensor लेता है और पहला output एक lazy
`Tensor` के रूप में लौटाता है। नतीजा realize होने पर कर्नेल चलता है, किसी भी graph kernel की तरह।

```rust
use svod_dtype::{DType, default_device::default_device};
use svod_tensor::Tensor;
use svod_tk3::atoms::Target;
use svod_tk3::build::BF16;
use svod_tk3::kernels::Batch;
use svod_tk3::kernels::gemm::{Epilogue, GemmCfg, GemmSpec, gemm};
use svod_tk3::launch::graph_launch;

let (m, n, k) = (1000, 512, 256);
let target = Target::for_device(&default_device()).expect("a GPU target");
let cfg = GemmCfg { tile: [64, 64, 32], stages: 3, warps: [2, 2], group_m: 8, unroll: true };
let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg };
let a = Tensor::empty(&[m * k], DType::BFloat16);
let b = Tensor::empty(&[n * k], DType::BFloat16);
let c = Tensor::empty(&[m * n], DType::BFloat16);
let c = graph_launch(gemm::<BF16>(&spec), &cfg.lowering(target), &[&a, &b, &c])?;
c.realize()?;
```

मॉडल यह ख़ुद नहीं करते: [`ops::linear`](./op-layer) config चुनता है, output को shape देता है और
कर्नेल लागू न होने पर ग्राफ़ पर लौट जाता है।

## Builder कॉल्स {#builder-calls}

| समूह | कॉल्स |
|---|---|
| घोषणाएँ | `param`, `var` (launch variable जो नाम से bind होता है), `grid`, `warps`, `smem` |
| Scalars | `block(axis)`, `warp()`, `load_scalar(param, index)`, `Sc` operators और `min`/`max`/`lt`/`le`/`eq`/`and`/`or` |
| Views | `view`, `at`, `smem_view`, `smem_slot` |
| Tile ops | `fill`, `zeros`, `splat`, `coord`, `unary`, `binary` (row या column vector broadcast होता है), `compare`, `cast`, `where_`, `reduce`, `mma` |
| Movement | `stage` (global → shared, `CopyMode::Async` या `Sync`), `load`, `store` |
| Control | `loop_` (carried register tiles), `pipeline`, `if_`, `select_if` (tiles बनाने वाली branches) |

`role_block`, `raw` और `transpose` रिकॉर्ड किए जा सकते हैं पर अभी lower नहीं होते: lowering उनके
लिए `Error::Unsupported` लौटाती है।
