---
sidebar_label: Your First Kernel
---

# Your First Kernel

A tk kernel is a Rust function that records a tile program with the `Kernel` builder:
`Kernel::record(name, |k| …)` opens the kernel and returns its `Program`. Every call appends
one statement to the open block and returns a typed handle. A handle carries its tier
(`Gmem<T>`, `Shared<T>`, `Regs<T>`), element type (`BF16`, `F16`, `F32`, `I32`, `Bool`) and
its kernel's brand, a lifetime no other kernel has, so a handle used in another kernel or a
pipeline slot kept past its stage is a compile error; a shared allocation is a `SmemRef<T>`.
Shapes are checked when the statement is recorded. Scalars such as block indices, offsets and
trip counts are `Sc` expressions built with ordinary operators, branded too.

## A minimal kernel

`y = 2·x + 1` over a `[rows, 64]` f32 matrix, 16 rows per block. The view's row bound makes the
last block's rows past `rows` read as zero and drops their writes, so `rows` need not be a
multiple of 16.

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

`view(param, offset, [row_stride, col_stride], shape, bounds)` is a window into a parameter.
`at(view, rows, cols)` moves it and keeps the bounds relative to the new origin.

## A GEMM

`c = a·bᵀ` over bf16 `a [m, k]` and `b [n, k]`, accumulated in f32. This is a reduced form of
`kernels/gemm.rs`, which adds the epilogues (bias, activation, gated weights, scale, residual,
f32 output), grouped tile order, split reductions, the warpgroup core and two-crew ping-pong.

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
            stages,
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

**Bounds instead of padding.** `bound(len, tile)` is `Some(len)` only where the tile grid
overhangs `len`, so any `m` and `n` run without a copy. Only `k` must be a multiple of `bk`.

**The pipeline.** `pipeline(extent, stages, init, produce, consume)` declares a producer and a
consumer over a ring of `stages` shared slots (a `Schedule::ring(stages, unroll)` also copies
the loop once per slot or runs two crews), carrying `init` (here the accumulator) through
the consumer. The author says what to copy into a slot and what to compute from it. The
[schedule template](./layouts-and-lowering#schedule-templates) decides how far ahead copies
run and places every wait and barrier. A slot is the stage's that holds it: `slot` gives the
producer a writable tile and the consumer a read-only one.

**The product.** `mma(acc, a, a_t, b, b_t)` is `acc + A·B` on f32 accumulators (i32 over 8-bit
integers, `MmaIn::Acc`), here with `B`
read transposed. It never names an instruction: the lowering picks the target's matrix-core
atom and gives `a`, `b` and `acc` its layouts. An accumulator (`Acc`) is owned, not copied:
the product consumes it and returns the next, a loop body returns its carried ones, and a
stale one is a compile error; `*acc` is the register tile it holds, for the epilogue. The
result is rounded once, at the cast before the store.

## Running it on the device

`launch::graph_launch` takes one tensor per declared parameter, in order, and returns the first
output as a lazy `Tensor`. The kernel runs when the result is realized, like any graph kernel.
The library GEMM takes its config as an `Admitted<GemmCfg>`: only the planner's admission (feasible on the
target, claimed on the shape) mints one, so no kernel is built from a config nothing admitted; its `lowering`
needs the device's measured costs.

```rust
use svod_dtype::{DType, default_device::default_device};
use svod_tensor::Tensor;
use svod_tk::atoms::{Device, Target};
use svod_tk::build::BF16;
use svod_tk::kernels::Batch;
use svod_tk::kernels::gemm::{Epilogue, GemmSpec, gemm};
use svod_tk::launch::graph_launch;
use svod_tk::ops::config::{GemmProblem, Planner};

let (m, n, k) = (1000, 512, 256);
let target = Target::for_device(&default_device()).expect("a GPU target");
let device = Device { costs: svod_tk::probe::costs(&target)?.costs, target };
let planner = Planner::new(device).expect("a target with an SM count");
let problem = GemmProblem { batches: 1, m, n, k, halves: 1, rope: None };
let ranked = planner.gemm_ranked(1, m, n, k, false);
let cfg = problem.admitted(&planner, ranked[0]).expect("the ranking admitted it");
let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg };
let a = Tensor::empty(&[m * k], DType::BFloat16);
let b = Tensor::empty(&[n * k], DType::BFloat16);
let c = Tensor::empty(&[m * n], DType::BFloat16);
let c = graph_launch(gemm::<BF16>(&spec), &cfg.lowering(&planner.device), &[&a, &b, &c])?;
c.realize()?;
```

Models do not do this themselves: [`ops::linear`](./op-layer) picks the config, shapes the
output and falls back to the graph when the kernel does not apply.

## Builder calls

| Group | Calls |
|---|---|
| Declarations | `input`, `output`, `inout`, `var` (a launch variable bound by name), `grid`, `warps` (a count, a column of warps, or a `WarpGrid` the register tiles spread over), `smem` |
| Scalars | `block(axis)`, `warp()`, `row()` (inside a gather map), `load_scalar(param, index)`, `Sc` operators and `min`/`max`/`lt`/`le`/`eq`/`and`/`or` |
| Views | `view`, `gather` (a row-mapped view), `periodic` (columns repeating a table), `at`, `narrow`, `smem_view`, `smem_slot`, `ring`, `slot`, `ring_scratch` |
| Tile ops | `fill`, `zeros`, `splat`, `coord`, `unary`, `binary` (a row or column vector broadcasts), `scale`, `compare` (`CmpOp`), `cast`, `cast_toward_zero`, `where_`, `mul_add` (one rounding), `reduce`, `reduce_f32`, `transpose`, `shift` (one step along an axis, a fill first), `flip` (each element exchanged across a mask's bits along an axis: a sorting network's step, a rotary half's partner), `mma`, `mma_oriented`, `pin` |
| Movement | `stage` (global → shared, `CopyMode::Async` or `Sync`), `load`, `load_as_fill`, `store` |
| Control | `loop_` (owned accumulators carried through the body; `loop_vec` carries plain tiles), `pipeline`, `if_`, `select_if` (branches producing tiles), `where_any` (a `where_` whose taken side only the warps holding a set predicate element compute), `each_warp` (a `Warps` statement), `barrier` |
