---
sidebar_label: 第一个内核
---

# 第一个内核 {#your-first-kernel}

tk 内核是一个 Rust 函数，它用 `Kernel` 构建器记录一个 tile 程序。每次调用都会向当前打开的块追加一条语句，并返回一个带类型的句柄。句柄携带其层级（`Gmem<T>`、`Shared<T>`、`Regs<T>`）和元素类型（`BF16`、`F16`、`F32`、`I32`、`Bool`）；共享内存分配是一个 `SmemRef<T>`。形状在语句被记录时检查。块索引、偏移量和迭代次数等标量是用普通运算符构造的 `Sc` 表达式。

## 一个最小的内核 {#a-minimal-kernel}

在 `[rows, 64]` 的 f32 矩阵上计算 `y = 2·x + 1`，每个块处理 16 行。视图的行边界使最后一个块中超出 `rows` 的行读为零、写入被丢弃，因此 `rows` 不必是 16 的倍数。

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

`view(param, offset, [row_stride, col_stride], shape, bounds)` 是参数上的一个窗口。`at(view, rows, cols)` 移动这个窗口，并让边界保持相对于新原点。

## 一个 GEMM {#a-gemm}

在 bf16 的 `a [m, k]` 和 `b [n, k]` 上计算 `c = a·bᵀ`，以 f32 累加。这是 `kernels/gemm.rs` 的简化形式；完整版本还加入了尾处理（偏置、激活、门控权重、缩放、残差、f32 输出）、分组 tile 顺序、拆分归约、warpgroup 核心以及双工作组乒乓。

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

**用边界代替填充。** 只有在 tile 网格超出 `len` 时，`bound(len, tile)` 才是 `Some(len)`，因此任意 `m` 和 `n` 都无需拷贝即可运行。只有 `k` 必须是 `bk` 的倍数。

**流水线。** `pipeline(extent, stages, init, produce, consume)` 在由 `stages` 个共享槽组成的环上声明一个生产者和一个消费者，并让 `init`（这里是累加器）贯穿消费者传递。作者只说明要把什么拷贝进槽、以及从槽中计算什么。[调度模板](./layouts-and-lowering#schedule-templates)决定拷贝提前多少步，并放置每一个等待和屏障。

**乘积。** `mma(acc, a, a_t, b, b_t)` 在 f32 累加器上计算 `acc + A·B`，这里 `B` 以转置方式读取。它从不指名某条指令：降级会选择目标的矩阵核心原子，并为 `a`、`b` 和 `acc` 赋予该原子的布局。结果只在存储前的类型转换处舍入一次。

## 在设备上运行 {#running-it-on-the-device}

`launch::graph_launch` 按顺序为每个声明的参数接收一个张量，并以惰性 `Tensor` 的形式返回第一个输出。与任何图内核一样，内核在结果被实现（realize）时运行。库中的 GEMM 接收一个 `GemmCfg`，其 `lowering` 需要设备测得的代价。

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
    GemmCfg { tile: [64, 64, 32], stages: 2, warps: [2, 2], group_m: 8, unroll: true, core: Core::Warp, crews: 1 };
let spec = GemmSpec { m, n, k, batch: Batch::Static(1), epilogue: Epilogue::default(), cfg };
let a = Tensor::empty(&[m * k], DType::BFloat16);
let b = Tensor::empty(&[n * k], DType::BFloat16);
let c = Tensor::empty(&[m * n], DType::BFloat16);
let c = graph_launch(gemm::<BF16>(&spec), &cfg.lowering(&device), &[&a, &b, &c])?;
c.realize()?;
```

模型自己不做这些：[`ops::linear`](./op-layer) 会选择配置、确定输出形状，并在内核不适用时回退到计算图。

## 构建器调用 {#builder-calls}

| 分组 | 调用 |
|---|---|
| 声明 | `param`、`var`（按名称绑定的启动变量）、`grid`、`warps`、`smem` |
| 标量 | `block(axis)`、`warp()`、`row()`（在 gather 映射内部）、`load_scalar(param, index)`、`Sc` 运算符以及 `min`/`max`/`lt`/`le`/`eq`/`and`/`or` |
| 视图 | `view`、`gather`（按行映射的视图）、`at`、`narrow`、`smem_view`、`smem_view_as`、`smem_slot` |
| Tile 操作 | `fill`、`zeros`、`splat`、`coord`、`unary`、`binary`（行向量或列向量会广播）、`scale`、`compare`（`CmpOp`）、`cast`、`cast_toward_zero`、`where_`、`mul_add`（只舍入一次）、`reduce`、`reduce_f32`、`transpose`、`mma`、`mma_oriented`、`pin` |
| 数据移动 | `stage`（全局 → 共享，`CopyMode::Async` 或 `Sync`）、`load`、`load_as_fill`、`store` |
| 控制 | `loop_`（携带寄存器 tile）、`pipeline`、`if_`、`select_if`（产生 tile 的分支）、`where_any`（一个 `where_`，只有持有已置位谓词元素的 warp 计算其选中一侧）、`each_warp`（一条 `Warps` 语句）、`barrier` |
