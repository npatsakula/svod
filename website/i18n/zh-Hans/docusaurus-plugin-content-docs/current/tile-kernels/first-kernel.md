---
sidebar_label: 第一个内核
---

# 第一个内核

tk3 内核是一个 Rust 函数，它用 `Kernel` 构建器记录一个 tile 程序。每次调用都会向当前打开的块追加一条语句，并返回一个带类型的句柄。句柄携带其层级（`Gmem<T>`、`Shared<T>`、`Regs<T>`）和元素类型（`BF16`、`F16`、`F32`、`I32`、`Bool`）。形状在语句被记录时检查。块索引、偏移量和迭代次数等标量是用普通运算符构造的 `Sc` 表达式。

## 一个最小的内核 {#a-minimal-kernel}

在 `[rows, 64]` 的 f32 矩阵上计算 `y = 2·x + 1`，每个块处理 16 行。视图的行边界使最后一个块中超出 `rows` 的行读为零、写入被丢弃，因此 `rows` 不必是 16 的倍数。

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

`view(param, offset, [row_stride, col_stride], shape, bounds)` 是参数上的一个窗口。`at(view, rows, cols)` 移动该窗口，并让边界保持相对于新原点。

## GEMM 逐步讲解 {#the-gemm-step-by-step}

`kernels/gemm.rs` 计算 `c = act(a·bᵀ + bias) + residual`（门控权重时为 `act(gate)·up`），连同 spec 和配置类型在内约 180 行。下面的摘录逐字摘自源码。

**参数与网格。** 缓冲区是扁平的，大小按批次容量分配。绑定的批次变量成为网格的 z 维。

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

**用边界代替填充。** 只有在 tile 网格超出 `len` 时，`bound(len, tile)` 才是 `Some(len)`。任意 `m` 和 `n` 都无需拷贝即可运行，只有 `k` 必须是 `bk` 的倍数。

```rust
let (bx, by) = tile_order(&mut k, gm, gn, cfg.group_m);
let (row0, col0) = (bx * bm, by * bn);
let (m_bound, n_bound) = (bound(m, bm), bound(n, bn));
let a_view = k.view(a, batch_offset(&bb, m * kk), [kk, 1], Shape::new(bm, bk), [m_bound.clone(), None]);
let a_view = k.at(a_view, row0.clone(), 0);
```

`tile_order` 按 `group_m` 个 tile 行为一组遍历，使驻留的块在 L2 中共享 B。

**流水线。** `pipeline(extent, stages, init, produce, consume)` 在由 `stages` 个共享槽组成的环上声明一个生产者和一个消费者。作者只说明要把什么拷贝进槽、以及从槽中计算什么。[调度模板](./layouts-and-lowering#schedule-templates)决定拷贝提前多少步，并放置每一个等待和屏障。

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

`mma(acc, a, a_t, b, b_t)` 在 f32 累加器上计算 `acc + A·B`。它从不指名具体指令：降级会选取目标的矩阵核心原子，并把其布局赋给 `a`、`b` 和 `acc`。

**尾处理（epilogue）** 在 f32 累加器上运行，结果在存储时只舍入一次。偏置是 `[1, bn]` 行向量，在整个 tile 上广播；残差是一个完整的 tile。

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

## 在设备上运行 {#running-it-on-the-device}

`launch::graph_launch` 按顺序为每个声明的参数接收一个张量，并把第一个输出作为惰性 `Tensor` 返回。与任何图内核一样，内核在结果被 realize 时运行。

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

模型不会自己这样做：[`ops::linear`](./op-layer) 会选择配置、确定输出形状，并在内核不适用时回退到计算图。

## 构建器调用 {#builder-calls}

| 分组 | 调用 |
|---|---|
| 声明 | `param`、`var`（按名称绑定的启动变量）、`grid`、`warps`、`smem` |
| 标量 | `block(axis)`、`warp()`、`load_scalar(param, index)`、`Sc` 运算符以及 `min`/`max`/`lt`/`le`/`eq`/`and`/`or` |
| 视图 | `view`、`at`、`smem_view`、`smem_slot` |
| Tile 操作 | `fill`、`zeros`、`splat`、`coord`、`unary`、`binary`（行向量或列向量会广播）、`compare`、`cast`、`where_`、`reduce`、`mma` |
| 数据移动 | `stage`（全局 → 共享，`CopyMode::Async` 或 `Sync`）、`load`、`store` |
| 控制 | `loop_`（携带寄存器 tile）、`pipeline`、`if_`、`select_if`（产生 tile 的分支） |

`role_block`、`raw` 和 `transpose` 可以被记录，但尚未被降级：降级对它们返回 `Error::Unsupported`。
