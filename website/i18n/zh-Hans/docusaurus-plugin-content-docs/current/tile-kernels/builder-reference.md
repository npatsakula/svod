---
sidebar_label: 构建器 API
---

# 构建器 API

[编写一个内核](./first-kernel) 只用到了寥寥几个调用。本页是 `svod-tk` 完整的 AUTHOR 面孔：编写内核体所用的每个类型与方法，按用途分组。除非给出了模块路径，这里的一切都从 `tk/src/lib.rs` 重新导出。

内核体是一个闭包 `FnOnce(&Kernel) -> Arc<UOp>`。它绑定全局、分配 tile、经一个 `Group` 发射 tile 操作，最后返回 `ker.finish(n)`。

---

## `Kernel`：上下文

```rust
// tk/src/kernel.rs
pub fn new(name: impl Into<String>, grid: [i64; 3], block: i64, buffers: Vec<Arc<UOp>>, caps: ArchCaps) -> Kernel
```

你很少亲手构造它：`run_kernel`、`compile_kernel`、`graph_launch` 和 `graph_launch_multi` 会替你构建，并绑定到启动缓冲区上。你会读取的字段：

| 字段 / 方法 | 含义 |
|---|---|
| `ker.caps` | 目标平台的 `ArchCaps`：`arch` 与 `wave_size`（CDNA 上为 64，RDNA、CUDA 和 Metal 上为 32） |
| `ker.grid_x()` / `grid_y()` / `grid_z()` | 以 `Special` UOp 表示的 `blockIdx.{x,y,z}`（仅在被使用时渲染） |
| `ker.thread_idx` | `threadIdx.x` |
| `ker.warpid()` / `ker.laneid()` | `threadIdx / wave_size` 与 `threadIdx % wave_size` |
| `ker.frag(role)` | 该架构上某个 `FragRole` 对应的物理 `RTBaseShape`；架构没有矩阵核心布局时 panic |

`block` 必须是整数个 wave（`Kernel::new` 中的 `debug_assert`）；启动块通常是 `warps * caps.wave_size`。

### 绑定全局

```rust
// tk/src/scaffold.rs
pub fn bind_abi(&self, outputs: &[GlSpec], inputs: &[GlSpec]) -> (Vec<GL>, Vec<GL>)
pub fn gl(&self, shape: &[usize], dtype: DType) -> GL                // tk/src/tile.rs
```

`bind_abi` 就是按切片顺序调用 `gl`：先输出，后输入，与交给启动器的缓冲区顺序一致。以所绑定缓冲区的 dtype 为准；debug 构建会断言所声明 dtype 的字节宽度与之相同。可选缓冲区（FA 的 `key_lens`）在 `bind_abi` 之后以一次追加的 `gl` 绑定，绝不能夹在中间。

```rust
let (outs, ins) = ker.bind_abi(
    &[GlSpec::new(&[1, 1, m, n], DType::BFloat16)],
    &[GlSpec::new(&[1, 1, m, k], DType::BFloat16), GlSpec::new(&[1, 1, n, k], DType::BFloat16)],
);
```

### 分配 tile

形状描述符（`tk/src/tiles.rs`）是纯数据；包装器（`tk/src/tile.rs`）则绑定一个缓冲区。原始构造函数接收一个显式的基础形状；脚手架快捷方式经 `ker.caps` 按角色解析它，树内内核用的正是后者。

| 原始 | 快捷方式 | 分配 |
|---|---|---|
| `ker.rt(dims, dtype, layout, base)` | `ker.acc(dims, layout)` | `Accumulator` 片段中的 f32 `RT` |
| | `ker.acc_t(dims, layout)` | `AccumulatorT` 中的 f32 `RT`（转置累加器的 N 主序存储） |
| | `ker.operand(dims, dt, layout)` | A 操作数片段中的 16 位 `RT` |
| | `ker.operand_b(dims, dt, layout)` | B 操作数片段（仅在 Metal 上与 `operand` 不同） |
| `ker.rv(length, dtype, VecLayout::Ortho, base)` | `ker.acc_vec(length)` | f32 `RV`，`length / frag_rows` 个 tile × `LaneMap::slots()` |
| `ker.st(dims, dtype, layout, base)` | `ker.shared(dims, dt, layout)` | 采用该架构普通条带布局的 LDS `ST` |
| | `ker.shared_sw(dims, dt, layout)` | 采用 XOR swizzle 条带布局的 LDS `ST` |
| `ker.st_db(..)` / `ker.st_stages(.., stages)` | `ker.shared_db(..)` / `ker.shared_sw_stages(.., stages)` | 同一个 tile，建在 `stages` 倍的缓冲区上，供软件流水线使用 |

`dims` 是以元素计的 `(rows, cols)`，两个轴上都必须是基础片段的整数倍（一个 `assert`）。`TileLayout::{Row, Col}` 说明一个 lane 的寄存器沿哪个轴排布；规约以及与全局内存之间的搬运都会读取它。

`RT` 的逻辑形状是 `[height, width, ept]`（片段网格，然后是每 lane 元素数）；`ST` 的是 `[height, width, frag_rows, frag_cols]`。`ST::subtile(dims, (row_blk, col_blk))` 是共享 tile 中某个 wave 那一条带的零拷贝视图；`ST::with_base_offset(off)` 选择一个流水线级（`parity * st.half_elems()`）。

### 次序

tile 是不可变句柄。每个操作返回的目标 tile 都被**重新包装**，带上一条指向它所发射存储的 `After` 边，于是下一次读取会排在它之后。数据流表达不了的地方，有两种手工串接的边：

- `t.after(deps)`：让 `t` 的下一次读取排在 `deps` 之后（`deps` 可以是 tile、range、barrier，或它们组成的数组或元组；见 `tk/src/tile.rs` 中的 `AfterDeps`）。
- `st.after(deps)`：`ST` 的对应版本。

---

## `Group`：计算词汇

```rust
ker.warp()             // 1 wave
ker.group(n)           // 1×n waves, for collaborative GLOBAL→LDS fills
ker.group_2d(r, c)     // an r×c wave grid; group_threads = r·c·wave_size
```

`g.warp_row()` / `g.warp_col()` 是该 wave 在网格中的坐标；`g.warpid_in_group()` 是它的扁平索引。寄存器操作是逐 lane 的，在任何组上都对 wave 安全；单 wave 操作（`map_position`、基于 `col_reduce` 的规约、混洗）会断言 `warps == 1`，即便在多 wave 内核中也要在 `ker.warp()` 上调用它们。

### 数据搬运

```rust
// tk/src/group/movement.rs
pub fn load<Dst, Src: LoadInto<'k, Dst>>(&self, dst: Dst, src: Src, ix: MoveIdx) -> Src::Output
pub fn store<Dst, Src: StoreInto<'k, Dst>>(&self, dst: Dst, src: Src, ix: MoveIdx) -> Src::Output
```

合法的地址空间组合以 trait impl 的形式给出，所以非法组合（`RT ← RT`）是编译错误：

| 调用 | 组合 | 发射什么 |
|---|---|---|
| `g.load(st, gl, ix)` | `ST ← GL` | 由组内全部线程完成的合并填充 + 一个工作组 barrier |
| `g.load(rt, st, ix)` | `RT ← ST` | 经 `LaneMap` 的逐 lane 片段 gather（CUDA 上每个 16 位片段一条 `ldmatrix.x4`） |
| `g.load(rt, gl, ix)` | `RT ← GL` | 直接的全局 gather，不经 LDS |
| `g.store(st, rt, ix)` | `ST ← RT` | 片段 scatter 到 LDS |
| `g.store(gl, rt, ix)` | `GL ← RT` | 片段 scatter 到全局 |

`MoveIdx` 按角色为索引命名：`MoveIdx::block(idxs, axis)` 是 tile 在全局中的坐标（每个全局维度一项；`axis` 是 tile 的一行所跨越其 stride 的那个维度），`MoveIdx::frag(idxs)` 是寄存器侧的片段偏移，`MoveIdx::at(block, frag, axis)` 两者兼有，`MoveIdx::default()` 两者皆无（子 tile 已自带其条带）。`.masked()` 让一次 `GLOBAL ↔ REG` 搬运受张量范围的门控：参差不齐的边缘读到 `0.0`，写入则被丢弃。

流水线原语把填充与它的同步拆开：

| 原语 | 架构 | 用途 |
|---|---|---|
| `fill_local_nobar` / `fill_local_vec_nobar` | 全部 | 不带尾随 barrier 的填充；由调用方设栅栏 |
| `stage_global_to_reg(st, gl, idxs, axis)` → `commit_regs_to_local(&[(st, stage), ..])` | 全部（AMD 路径） | 现在把全局加载读入寄存器，稍后再 `ds_write` 进 LDS，让加载在当前块的 MMA 期间保持在途 |
| `cp_async_fill(st, gl, idxs, axis)`（受 `cp_async_fill_applies` 门控） | CUDA sm_80+ | 16 字节 `cp.async` 直接写入 LDS；用 `cp_async_wait(n, ..)` + `.barrier(..)` 收尾 |
| `war_fence2(a, b, extra)` | 全部 | 两次 gather 都消费的跨 wave barrier，把预取提交作为依赖携带 |
| `store_local_fenced(st, rt, ix, deps)` | 全部 | 一次 `RT → ST` scatter 后接一个 barrier（RDNA3 的 softmax 重布局） |
| `store_global_with(gl, rt, ix, f)` | 全部 | 值为 `f(v, offset)` 的全局存储，即融合 epilogue |

### 矩阵乘法

```rust
// tk/src/group/mma.rs — C += A·B over every output fragment, reducing along K
pub fn mma_ab  (&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>   // a[h, k] · b[k, w]
pub fn mma_abt (&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>   // a[h, k] · b[w, k]ᵀ
pub fn mma_atb (&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>   // a[k, h]ᵀ · b[k, w]
pub fn mma_atbt(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>
```

`a`/`b` 是位于操作数片段中的 bf16 或 f16，`c` 是位于累加器片段中的 f32（否则 panic）。AMD 上每个 16×16×16 步一个 `Op::Wmma`，CUDA 上两个 `m16n8k16`，Apple 上每个片段一个 8×8×8 `simdgroup_matrix` 操作；描述符来自调度器的 `TensorCore` 表，因此手写内核与 BEAM 的 `TC` 动作共用同一个来源。

### 规约与混洗

```rust
// tk/src/group/reduce.rs
pub fn row_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init: f64) -> RV<'k>
pub fn col_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init: f64) -> RV<'k>
pub fn arg_reduce(&self, val: RV<'k>, idx: RV<'k>, src: &RT<'k>, dir: ArgDir) -> (RV<'k>, RV<'k>)
```

规约先折叠 lane 本地的元素，再借助片段的 `ReduceTree` 跨 lane 完成：AMD 上是 `ds_bpermute` 兄弟 gather，CUDA 与 Metal 上是 `shfl.bfly` 蝶形。`op` 是任意满足结合律的组合函数（`|a, b| a.max(b)`、`|a, b| a.add(b)`）；结果折入 `vec`，所以 `vec` 必须已持有当前的滚动值。

标量 wave 原语（`tk/src/group/shuffle.rs`）：`wave_reduce_scalar(value, op)`、`subgroup_reduce_scalar(value, width, op)`、`broadcast_scalar(value, lane)`，以及 tile 形式的 `shuffle_xor`、`shuffle_down`、`shuffle_up`、`compare_exchange`（双调排序的各级）。它们都不碰 LDS。

### 逐元素

| 调用 | 含义 |
|---|---|
| `g.zero(rt)` / `g.ones(rt)` / `g.neg_inf(rt)`；`zero_rv` / `clear_rv(rv, v)` | 常量填充 |
| `g.copy(dst, &src)` | 逐元素拷贝，dtype 不符时做类型转换 |
| `g.transpose(dst, &src)` | 交换片段网格的 `height` 与 `width` |
| `g.map(t, \|x, idx\| ..)` | 对每个元素应用一个 UOp 表达式 |
| `g.map_position(rt, row_blk, col_blk, \|x, idx, row, col\| ..)` | 同上，另附该元素从 `LaneMap` 读出的全局 `(row, col)` |
| `g.mask_where(rt, row_blk, col_blk, fill, \|row, col\| pred)` | `where(pred, fill, x)`，即因果掩码与填充掩码 |
| `g.add/sub/mul/div/maximum(a, &b)`、`*_scalar(a, s)`、`*_rv(rt, &rv)`、`g.exp2(t)` | tile 数学（`tk/src/math.rs`） |

运算符语法糖（`tk/src/ops.rs`）路由到同样的调用，于是内核体读起来就像数学。`tk/src/kernels/fa.rs` 中的在线 softmax 更新是：

```rust
let scale_vec = (max_vec_last - &max_vec).exp2();
o_reg = o_reg * &scale_vec;
norm_vec = norm_vec * &scale_vec;
let att = (att - &max_vec).exp2();
```

`T op &T` 要求形状相同，`RT op &RV` 沿 tile 的布局轴广播向量，`T op f64` 则是标量运算。

---

## 循环

```rust
// tk/src/loop_scope.rs
let lp = ker.loop_static(trips);          // a tracked RANGE with a constant trip count
let lp = ker.loop_dynamic(bound_uop);     // a runtime trip count (FA's causal block-skip)
lp.index()                                 // the counter, for addressing
lp.reinit(t)                               // t.after(range): re-run a per-trip init inside the loop
lp.close()                                 // end the last store around the range; returns the END
lp.close_carry(t)                          // close and rebind one carried tile to its post-loop value
lp.close_barrier(commits)                  // close with a workgroup fence folded into the END
```

循环作用域的存在，就是为了让以下两条规则无法被遗忘：

- 每轮迭代的重新初始化（内核体开头的 `g.zero(acc)`）必须依赖循环计数器，否则线性化器会把它提升到循环之上，累加器就会带着陈旧状态。请写成 `g.zero(lp.reinit(acc))`。
- 一个 `RANGE` 只接受恰好一个 `END`。一个循环里有多个累加器时，把其余的串进那唯一一个闭合存储（GEMM 让每个累加器的 A 输入穿过前一个累加器的 MMA），再以 `acc.after(&ended)` 读取各自的最终值。

`Kernel::range` / `range_uop` / `endrange` / `endrange_to` / `endrange_barrier_to` 是 `Loop` 所包装的原始形式；它们发射完全相同的图。

---

## 收尾与启动

```rust
pub fn finish(&self, stores: usize) -> Arc<UOp>          // tk/src/kernel.rs
```

`finish(n)` 弹出最后 `n` 个终结存储（每个输出全局一个），在每个外面关闭仍处于打开状态的被跟踪 range，并以 `KernelInfo { opts_to_apply: Some(vec![]), name: Some(name) }` 把它们汇入 sink。在 `finish` 时仍留有打开 range 的内核必须只有一个存储。存储经搬运操作进入栈，或通过 `ker.push_store(store, buf)` 显式压入（直线式的 norm 内核就是这样归并其向量存储的）。

```rust
// tk/src/launch.rs
pub fn graph_launch(name, grid, block, out: Tensor, ins: &[&Tensor], caps: ArchCaps, build) -> Result<Tensor>
pub fn graph_launch_multi(name, grid, block, outs: Vec<Tensor>, ins, caps, build) -> Result<Vec<Tensor>>
pub fn launch_custom<T>(device, archs: ArchSet, validate, applies, build) -> Result<Option<T>>
pub fn run_kernel(name, grid, block, outs: &mut [&mut Tensor], ins: &[&Tensor], build) -> Result<()>
pub fn compile_kernel(name, grid, block, outs, ins, build) -> Result<CompiledLaunch>
```

`graph_launch` 把 `SINK` 包装成一个 `Op::Call` 节点并返回一个惰性张量；`out` 是 `Tensor::empty(shape, dtype)`，内核体看到的占位符是 `[out, ins...]`，即 `bind_abi` 的顺序。`launch_custom` 是每个库内核都遵循的三态策略（[向 IR 中编写](./lowering)）：先对照内核的 `ArchSet` 做 `resolve_supported_arch`（不在其中则 `Ok(None)`），再 `validate(arch)`（请求畸形则 `Err`），再 `applies(arch)`（形状无法分块则 `Ok(None)`），最后 `build(arch)`。

```rust
// tk/src/kernels/norm.rs — the shape of every graph-native entry
crate::launch_custom(
    &x.device(),
    NORM_SUPPORTED_ARCHS,
    move |_arch| check_norm_operands("rms-norm", &check.0, &check.1, &check.2, check.3),
    move |arch| select_norm_cfg(rows, d, crate::ArchCaps::for_arch(arch).wave_size).is_some(),
    move |arch| {
        let caps = crate::ArchCaps::for_arch(arch);
        let cfg = select_norm_cfg(rows, d, caps.wave_size).expect("checked by the fit predicate");
        let (grid, block) = launch_dims(rows, cfg.rows_per_block, caps.wave_size);
        let out = Tensor::empty(&xd, dtype.clone());
        let dt = dtype.clone();
        crate::graph_launch("rms_norm", grid, block, out, &[x, weight], caps, move |ker| {
            build_row_norm(ker, rows, d, dt, eps, cfg, false);
            ker.finish(1)
        })
    },
)
```

`run_kernel` / `compile_kernel` 是直接派发的 DEBUG 面孔；见 [调试](./debugging)。

---

## tile 之下

有些内核需要的是地址，而不是 tile。`tk/src/index.rs` 是每个 tile 操作赖以构建的扁平寻址层，而且它是公开的：`Idx`（`Const(i64)` 或 `Uop`）、`flat_index(buf,
shape, idxs)`、`load_at`、`load_off`、`load_off_vec(buf, off, w)` / `store_off_vec`（每 lane 一次 `w` 宽的访问，渲染器会把它折叠成一条 128 位指令），以及带门控的 `load_off_gated` / `index_off_gated`。RMS-norm 内核和 `model/src/qwen3/tk/mod.rs` 中的 Qwen3 QKV-norm-RoPE 序言完全写在这一层：每行一个 wave，没有 `RANGE`，没有 LDS，并复用 norm 的行词汇（`plan`、`vload`、`vpick`、`vstore`、`inv_rms`、`scale_by`）。

`tk/src/asm.rs` 把 AMD 机器调度器的控制以带类型的 `Op::Custom` 节点暴露出来，每个都串在一个依赖上：`s_setprio(prio, dep)`、`s_waitcnt_lgkmcnt(n, dep)`、`sched_barrier(mask, dep)`、`iglp_opt(mode, dep)`。GEMM 在 gfx12 上使用 `sched_barrier(0, ..)`，因为 `ArchCaps::needs_pipeline_commit_fence()` 表明，否则后端调度器会把流水线的 LDS 提交提升到本轮 MMA 之前。

`tk/src/grid.rs::l2_swizzle(wgid, num_wgs, grid_m, grid_n)` 把扁平化的工作组 id 映射为 `(pid_m, pid_n)`，让共同调度的工作组共享同一个 XCD 的 L2（HipKittens 的小芯片变换）；`GemmCfg::l2_swizzle` 用于开启它。
