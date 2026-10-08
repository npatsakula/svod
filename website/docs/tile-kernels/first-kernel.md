---
sidebar_label: Your First Kernel
---

# Your First Kernel

A tk3 kernel is a Rust function that records a tile program with the `Kernel` builder. Every
call appends one statement to the open block and returns a typed handle. A handle carries its
tier (`Gmem<T>`, `Shared<T>`, `Regs<T>`) and element type (`BF16`, `F16`, `F32`, `I32`,
`Bool`). Shapes are checked when the statement is recorded. Scalars such as block indices,
offsets and trip counts are `Sc` expressions built with ordinary operators.

## A minimal kernel

`y = 2·x + 1` over a `[rows, 64]` f32 matrix, 16 rows per block. The view's row bound makes the
last block's rows past `rows` read as zero and drops their writes, so `rows` need not be a
multiple of 16.

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

`view(param, offset, [row_stride, col_stride], shape, bounds)` is a window into a parameter.
`at(view, rows, cols)` moves it and keeps the bounds relative to the new origin.

## The GEMM, step by step

`kernels/gemm.rs` computes `c = act(a·bᵀ + bias) + residual` (or `act(gate)·up` off a gated
weight) in about 180 lines including its spec and config types. The excerpts below are verbatim.

**Parameters and grid.** Buffers are flat and sized at batch capacity. A bound batch variable
becomes grid z.

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

**Views with bounds instead of padding.** `bound(len, tile)` is `Some(len)` only where the tile
grid overhangs `len`. Any `m` and `n` run without a copy. Only `k` must be a multiple of `bk`.

```rust
let (bx, by) = tile_order(&mut k, gm, gn, cfg.group_m);
let (row0, col0) = (bx * bm, by * bn);
let (m_bound, n_bound) = (bound(m, bm), bound(n, bn));
let a_view = k.view(a, batch_offset(&bb, m * kk), [kk, 1], Shape::new(bm, bk), [m_bound.clone(), None]);
let a_view = k.at(a_view, row0.clone(), 0);
```

`tile_order` walks groups of `group_m` tile rows so resident blocks share B in L2.

**The pipeline.** `pipeline(extent, stages, init, produce, consume)` declares a producer and a
consumer over a ring of `stages` shared slots. The author says what to copy into a slot and
what to compute from it. The [schedule template](./layouts-and-lowering#schedule-templates)
decides how far ahead copies run and places every wait and barrier.

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

`mma(acc, a, a_t, b, b_t)` is `acc + A·B` on f32 accumulators. It never names an instruction:
the lowering picks the target's matrix-core atom and gives `a`, `b` and `acc` its layouts.

**The epilogue** runs on the f32 accumulator, and the result is rounded once at the store. The
bias is a `[1, bn]` row vector broadcast over the tile, and the residual is a full tile.

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

## Running it on the device

`launch::graph_launch` takes one tensor per declared parameter, in order, and returns the first
output as a lazy `Tensor`. The kernel runs when the result is realized, like any graph kernel.

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

Models do not do this themselves: [`ops::linear`](./op-layer) picks the config, shapes the
output and falls back to the graph when the kernel does not apply.

## Builder calls

| Group | Calls |
|---|---|
| Declarations | `param`, `var` (a launch variable bound by name), `grid`, `warps`, `smem` |
| Scalars | `block(axis)`, `warp()`, `load_scalar(param, index)`, `Sc` operators and `min`/`max`/`lt`/`le`/`eq`/`and`/`or` |
| Views | `view`, `at`, `smem_view`, `smem_slot` |
| Tile ops | `fill`, `zeros`, `splat`, `coord`, `unary`, `binary` (a row or column vector broadcasts), `compare`, `cast`, `where_`, `reduce`, `mma` |
| Movement | `stage` (global → shared, `CopyMode::Async` or `Sync`), `load`, `store` |
| Control | `loop_` (carried register tiles), `pipeline`, `if_`, `select_if` (branches producing tiles) |

`role_block`, `raw` and `transpose` can be recorded but are not lowered yet: the lowering
returns `Error::Unsupported` for them.
