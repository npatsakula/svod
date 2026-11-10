---
sidebar_label: 第一个内核
---

# 第一个内核 {#your-first-kernel}

tk 内核是一个 Rust 函数，它用 `Kernel` 构建器记录一个 tile 程序：`Kernel::record(name, |k| …)` 打开内核并返回其 `Program`，该程序在 crate 之外只读（`name()`、`params()`、`body()`）。每次调用都会向当前打开的块追加一条语句，并返回一个带类型的句柄。句柄携带其层级（`Gmem<T>`、`Shared<T>`、`Regs<T>`）、元素类型（`BF16`、`F16`、`I8`、`F32`、`I32`、`Bool`）以及所属内核的标记——一个其他内核都没有的生命周期——因此在另一个内核中使用句柄，或把流水线的槽位保留到其阶段之外，都是编译错误；共享内存分配是一个 `SmemRef<T>`。形状在语句被记录时检查。块索引、偏移量和迭代次数等标量是由内核的标量和整数通过普通运算符构造的 `Sc` 表达式，同样带有标记。`Sc` 是不透明的：当某个调用消耗它时，构建器把它驻留到程序中，任何标量 id 都不会离开内核。

## 一个最小的内核 {#a-minimal-kernel}

在 `[rows, 64]` 的 f32 矩阵上计算 `y = 2·x + 1`，每个块处理 16 行。视图的行边界使最后一个块中超出 `rows` 的行读为零、写入被丢弃，因此 `rows` 不必是 16 的倍数。

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

`view(param, offset, [row_stride, col_stride], shape, bounds)` 是参数上的一个窗口。`at(view, rows, cols)` 移动这个窗口，并让边界保持相对于新原点。

## 一个 GEMM {#a-gemm}

在 bf16 的 `a [m, k]` 和 `b [n, k]` 上计算 `c = a·bᵀ`，以 f32 累加。这是 `kernels/gemm.rs` 的简化形式；完整版本还加入了尾处理（偏置、激活、门控权重、缩放、残差、f32 输出）、分组 tile 顺序、拆分归约、warpgroup 核心以及双工作组乒乓。

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

**用边界代替填充。** 只有在 tile 网格超出 `len` 时，`bound(len, tile)` 才是 `Some(len)`，因此任意 `m` 和 `n` 都无需拷贝即可运行。只有 `k` 必须是 `bk` 的倍数。

**流水线。** `ring(name, slot, schedule)` 分配 `schedule` 所运行数量（`schedule.stages()`）的共享槽，每个槽 `slot` 个元素。像这里这样传入槽数，就是一个不展开的环；`Schedule::ring(stages, unroll)` 在 `unroll` 时按每个槽复制一次循环，`Schedule::ping_pong(kslice)` 则让两个工作组共用一个槽运行。`pipeline(extent, ring, init, produce, consume)` 按其环的调度运行：它在环的槽上声明一个生产者和一个消费者，并让 `init`（这里是累加器）贯穿消费者传递。其阶段访问的任何其他环都必须持有同样多的槽：越过其分配的共享视图会使降级失败（`lower::Error::SharedPast`）。作者只说明要把什么拷贝进槽、以及从槽中计算什么。[调度模板](./layouts-and-lowering#schedule-templates)决定拷贝提前多少步，并放置每一个等待和屏障。槽归持有它的阶段所有：`slot` 给生产者一个可写的瓦片，给消费者一个只读的瓦片。

**乘积。** `mma(acc, a, a_t, b, b_t)` 在 f32 累加器（8 位整数时为 i32，`MmaIn::Acc`）上计算 `acc + A·B`，这里 `B` 以转置方式读取。它从不指名某条指令：降级会选择目标的矩阵核心原子，并为 `a`、`b` 和 `acc` 赋予该原子的布局。累加器（`Acc`）不是 `Copy`：乘积消耗它并返回下一个，循环体返回它携带的累加器，因此把已被乘积消耗的累加器继续传下去是编译错误。这只是 lint 级别的检查，而非所有权保证：`*acc` 是它持有的寄存器 tile（供收尾阶段使用），任何寄存器 tile 都能通过 `into` 变成 `Acc`，所以在乘积之前保留的副本若再次变成累加器，就会携带过期的值，编译器和降级都不会拒绝。结果只在存储前的类型转换处舍入一次。

## 在设备上运行 {#running-it-on-the-device}

`launch::graph_launch` 按顺序为每个声明的参数接收一个张量，并以惰性 `Tensor` 的形式返回第一个输出。与任何图内核一样，内核在结果被实现（realize）时运行。库中的 GEMM 以 `Admitted<GemmSpec>` 同时接收其 spec 与配置。`GemmSpec::admitted(&planner, cfg)` 在规划器的准入通过时铸造它（配置在目标上对该 spec 的问题可行、在其形状上有声明）；只有 `unsafe` 的 `Admitted::assume(spec, cfg)` 能不经准入接收配置，且不做准入所做的任何检查。准入后的 spec 只读，其配置的 `lowering` 需要设备测得的代价。

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

模型自己不做这些：[`ops::linear`](./op-layer) 会选择配置、确定输出形状，并在内核不适用时回退到计算图。

## 构建器调用 {#builder-calls}

| 分组 | 调用 |
|---|---|
| 声明 | `input`、`output`、`inout`、`var`（按名称绑定的启动变量，范围非空）、`grid`、`warps`（一个数量、一列 warp，或寄存器 tile 铺开所依据的 `WarpGrid`；在 `lone_warps` 之前设置）、`lone_warps`（超出网格的 warp，只在 `each_warp` 内运行）、`smem` |
| 标量 | `block(axis)`、`warp()`、`row()`（在 gather 映射内部）、`load_scalar(param, index)`（整数参数，`I8` 或 `I32`）、`Sc` 运算符以及 `min`/`max`/`lt`/`le`/`eq`/`and`/`or` |
| 视图 | `view`、`gather`（按行映射的视图）、`periodic`（列重复一张表）、`at`、`narrow`、`smem_view`、`smem_slot`、`ring`、`ring_runs`（每个运行自己流水线的 warp 一组调度槽）、`slot`、`slot_part`（阶段槽的一部分）、`ring_scratch`（`unsafe`：在没有阶段访问的地方把环当作暂存区） |
| Tile 操作 | `fill`、`zeros`、`splat`、`coord`、`unary`、`binary`（行向量或列向量会广播）、`scale`、`compare`（`CmpOp`）、`cast`、`cast_toward_zero`、`where_`、`mul_add`（只舍入一次）、`reduce`、`reduce_f32`、`transpose`、`shift`（沿某轴移动一步，首位填充）、`flip`（每个元素沿某轴与掩码各位相异的元素交换：排序网络的一步、旋转编码中半部的配对）、`deal`/`deal_col`（每个 lane 持有的某一行的某一列及其列号）、`mma`、`mma_oriented`、`pin` |
| 数据移动 | `stage`（全局 → 共享，`CopyMode::Async` 或 `Sync`）、`load`、`load_as_fill`、`store` |
| 控制 | `loop_`（在循环体中携带累加器；`loop_vec` 携带普通 tile）、`pipeline`（`pipeline_vec` 携带普通 tile）、`if_`、`select_if`（产生 tile 的分支；`select_if_vec` 作用于 `Vec`）、`where_any`（一个 `where_`，只有持有已置位谓词元素的 warp 计算其选中一侧）、`if_any`（只有这些 warp 执行的分支，之后不做选择）、`each_warp`（一条 `Warps` 语句）、`barrier` |

只有在定义支配读取的地方才能读取 tile：从循环体、分支或 `each_warp` 中保留下来并在其外读取的 tile 会使降级失败（`lower::Error::Escaped`）。在谓词随 warp 变化的分支下（如 `where_any` 和 `if_any` 的分支）、部分 warp 会跳过的块屏障，同样会使降级失败（`lower::Error::DivergentBarrier`）。
