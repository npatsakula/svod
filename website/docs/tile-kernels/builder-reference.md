---
sidebar_label: The Builder API
---

# The Builder API

[Writing a Kernel](./first-kernel) used a handful of calls. This page is the full AUTHOR
face of `svod-tk`: every type and method a kernel body is written with, grouped by what it
does. Everything here is re-exported from `tk/src/lib.rs` unless a module path is given.

A kernel body is a closure `FnOnce(&Kernel) -> Arc<UOp>`. It binds globals, allocates tiles,
emits tile ops through a `Group`, and returns `ker.finish(n)`.

---

## `Kernel` — the context

```rust
// tk/src/kernel.rs
pub fn new(name: impl Into<String>, grid: [i64; 3], block: i64, buffers: Vec<Arc<UOp>>, caps: ArchCaps) -> Kernel
```

You rarely construct one: `run_kernel`, `compile_kernel`, `graph_launch` and
`graph_launch_multi` build it for you, bound to the launch buffers. The fields you read:

| Field / method | Meaning |
|---|---|
| `ker.caps` | the `ArchCaps` of the target: `arch` and `wave_size` (64 on CDNA, 32 on RDNA, CUDA and Metal) |
| `ker.grid_x()` / `grid_y()` / `grid_z()` | `blockIdx.{x,y,z}` as `Special` UOps (rendered only if used) |
| `ker.thread_idx` | `threadIdx.x` |
| `ker.warpid()` / `ker.laneid()` | `threadIdx / wave_size` and `threadIdx % wave_size` |
| `ker.frag(role)` | the physical `RTBaseShape` for a `FragRole` on this arch — panics where the arch has no matrix-core layouts |

`block` must be a whole number of waves (`debug_assert` in `Kernel::new`); the launch block
is normally `warps * caps.wave_size`.

### Binding globals

```rust
// tk/src/scaffold.rs
pub fn bind_abi(&self, outputs: &[GlSpec], inputs: &[GlSpec]) -> (Vec<GL>, Vec<GL>)
pub fn gl(&self, shape: &[usize], dtype: DType) -> GL                // tk/src/tile.rs
```

`bind_abi` is `gl` in slice order: outputs first, then inputs, matching the buffer order the
launcher was handed. The bound buffer's dtype governs; a debug build asserts the declared
dtype has the same byte width. An optional buffer (FA's `key_lens`) is bound with a trailing
`gl` after `bind_abi`, never between.

```rust
let (outs, ins) = ker.bind_abi(
    &[GlSpec::new(&[1, 1, m, n], DType::BFloat16)],
    &[GlSpec::new(&[1, 1, m, k], DType::BFloat16), GlSpec::new(&[1, 1, n, k], DType::BFloat16)],
);
```

### Allocating tiles

Shape descriptors (`tk/src/tiles.rs`) are pure data; the wrappers (`tk/src/tile.rs`) bind a
buffer. The raw constructors take an explicit base shape; the scaffold shortcuts resolve it
by role through `ker.caps` and are what the in-tree kernels use.

| Raw | Shortcut | Allocates |
|---|---|---|
| `ker.rt(dims, dtype, layout, base)` | `ker.acc(dims, layout)` | f32 `RT` in the `Accumulator` fragment |
| | `ker.acc_t(dims, layout)` | f32 `RT` in `AccumulatorT` (N-major store of a transposed accumulator) |
| | `ker.operand(dims, dt, layout)` | 16-bit `RT` in the A-operand fragment |
| | `ker.operand_b(dims, dt, layout)` | the B-operand fragment (differs from `operand` only on Metal) |
| `ker.rv(length, dtype, VecLayout::Ortho, base)` | `ker.acc_vec(length)` | f32 `RV`, `length / frag_rows` tiles × `LaneMap::slots()` |
| `ker.st(dims, dtype, layout, base)` | `ker.shared(dims, dt, layout)` | LDS `ST` in the arch's plain strip |
| | `ker.shared_sw(dims, dt, layout)` | LDS `ST` in the XOR-swizzled strip |
| `ker.st_db(..)` / `ker.st_stages(.., stages)` | `ker.shared_db(..)` / `ker.shared_sw_stages(.., stages)` | the same tile over a `stages`× buffer for a software pipeline |

`dims` is `(rows, cols)` in elements and must be a multiple of the base fragment on both axes
(an `assert`). `TileLayout::{Row, Col}` says which axis a lane's registers run along; the
reductions and the global hops read it.

An `RT`'s logical shape is `[height, width, ept]` (fragment grid, then elements per lane); an
`ST`'s is `[height, width, frag_rows, frag_cols]`. `ST::subtile(dims, (row_blk, col_blk))`
is a zero-copy view of one wave's band of a shared tile; `ST::with_base_offset(off)` selects a
pipeline stage (`parity * st.half_elems()`).

### Ordering

Tiles are immutable handles. Every op returns the destination tile **rewrapped** with an
`After` edge on the store it emitted, so the next read orders after it. Two hand-threaded
edges exist for what the dataflow does not express:

- `t.after(deps)` — order `t`'s next read after `deps` (a tile, a range, a barrier, an
  array or tuple of them; `AfterDeps` in `tk/src/tile.rs`).
- `st.after(deps)` — the `ST` analog.

---

## `Group` — the compute vocabulary

```rust
ker.warp()             // 1 wave
ker.group(n)           // 1×n waves, for collaborative GLOBAL→LDS fills
ker.group_2d(r, c)     // an r×c wave grid; group_threads = r·c·wave_size
```

`g.warp_row()` / `g.warp_col()` are the wave's coordinates in the grid; `g.warpid_in_group()`
its flat index. Register ops are per-lane and wave-safe on any group; the single-wave ops
(`map_position`, reductions over `col_reduce`, shuffles) assert `warps == 1` — call them on
`ker.warp()` even in a multi-wave kernel.

### Movement

```rust
// tk/src/group/movement.rs
pub fn load<Dst, Src: LoadInto<'k, Dst>>(&self, dst: Dst, src: Src, ix: MoveIdx) -> Src::Output
pub fn store<Dst, Src: StoreInto<'k, Dst>>(&self, dst: Dst, src: Src, ix: MoveIdx) -> Src::Output
```

The legal address-space pairs are trait impls, so an illegal pair (`RT ← RT`) is a compile
error:

| Call | Pair | What it emits |
|---|---|---|
| `g.load(st, gl, ix)` | `ST ← GL` | coalesced fill over all group threads + a workgroup barrier |
| `g.load(rt, st, ix)` | `RT ← ST` | per-lane fragment gather through the `LaneMap` (one `ldmatrix.x4` per 16-bit fragment on CUDA) |
| `g.load(rt, gl, ix)` | `RT ← GL` | direct global gather, no LDS stop |
| `g.store(st, rt, ix)` | `ST ← RT` | fragment scatter into LDS |
| `g.store(gl, rt, ix)` | `GL ← RT` | fragment scatter to global |

`MoveIdx` names the indices by role: `MoveIdx::block(idxs, axis)` is the tile's coordinate
in the global (one entry per global dim; `axis` is the dim whose stride a tile row spans),
`MoveIdx::frag(idxs)` a register-side fragment offset, `MoveIdx::at(block, frag, axis)` both,
`MoveIdx::default()` nothing (a subtile already carries its band). `.masked()` gates a
`GLOBAL ↔ REG` hop against the tensor's extent: a ragged edge reads `0.0` and drops the
write.

The pipeline primitives split a fill from its synchronization:

| Primitive | Arch | Use |
|---|---|---|
| `fill_local_nobar` / `fill_local_vec_nobar` | all | a fill with no trailing barrier; the caller fences |
| `stage_global_to_reg(st, gl, idxs, axis)` → `commit_regs_to_local(&[(st, stage), ..])` | all (the AMD path) | global loads into registers now, `ds_write` into LDS later, so the loads are in flight under the current block's MMAs |
| `cp_async_fill(st, gl, idxs, axis)` (gated by `cp_async_fill_applies`) | CUDA sm_80+ | 16-byte `cp.async` straight into LDS; retire with `cp_async_wait(n, ..)` + `.barrier(..)` |
| `war_fence2(a, b, extra)` | all | a cross-wave barrier both gathers consume, carrying the prefetch commits as deps |
| `store_local_fenced(st, rt, ix, deps)` | all | an `RT → ST` scatter followed by a barrier (the RDNA3 softmax relayout) |
| `store_global_with(gl, rt, ix, f)` | all | a global store whose value is `f(v, offset)` — the fused epilogues |

### Matrix multiply

```rust
// tk/src/group/mma.rs — C += A·B over every output fragment, reducing along K
pub fn mma_ab  (&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>   // a[h, k] · b[k, w]
pub fn mma_abt (&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>   // a[h, k] · b[w, k]ᵀ
pub fn mma_atb (&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>   // a[k, h]ᵀ · b[k, w]
pub fn mma_atbt(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k>
```

`a`/`b` are bf16 or f16 in the operand fragments, `c` f32 in the accumulator fragment
(panics otherwise). One `Op::Wmma` per 16×16×16 step on AMD, two `m16n8k16` on CUDA, one
8×8×8 `simdgroup_matrix` op per Apple fragment; the descriptor comes from the scheduler's
`TensorCore` table, so hand kernels and BEAM's `TC` action share one source.

### Reductions and shuffles

```rust
// tk/src/group/reduce.rs
pub fn row_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init: f64) -> RV<'k>
pub fn col_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init: f64) -> RV<'k>
pub fn arg_reduce(&self, val: RV<'k>, idx: RV<'k>, src: &RT<'k>, dir: ArgDir) -> (RV<'k>, RV<'k>)
```

A reduce folds the lane-local elements, then completes across lanes with the fragment's
`ReduceTree` — a `ds_bpermute` sibling gather on AMD, a `shfl.bfly` butterfly on CUDA and
Metal. `op` is any associative combiner (`|a, b| a.max(b)`, `|a, b| a.add(b)`); the result
folds into `vec`, so `vec` must already hold the running value.

Scalar wave primitives (`tk/src/group/shuffle.rs`): `wave_reduce_scalar(value, op)`,
`subgroup_reduce_scalar(value, width, op)`, `broadcast_scalar(value, lane)`, and the tile
forms `shuffle_xor`, `shuffle_down`, `shuffle_up`, `compare_exchange` (bitonic stages). None
of them touch LDS.

### Elementwise

| Call | Meaning |
|---|---|
| `g.zero(rt)` / `g.ones(rt)` / `g.neg_inf(rt)`; `zero_rv` / `clear_rv(rv, v)` | constant fills |
| `g.copy(dst, &src)` | element copy, casting on a dtype mismatch |
| `g.transpose(dst, &src)` | swap the fragment grid's `height` and `width` |
| `g.map(t, \|x, idx\| ..)` | apply a UOp expression per element |
| `g.map_position(rt, row_blk, col_blk, \|x, idx, row, col\| ..)` | the same with the element's global `(row, col)`, read off the `LaneMap` |
| `g.mask_where(rt, row_blk, col_blk, fill, \|row, col\| pred)` | `where(pred, fill, x)` — the causal and padding masks |
| `g.add/sub/mul/div/maximum(a, &b)`, `*_scalar(a, s)`, `*_rv(rt, &rv)`, `g.exp2(t)` | tile math (`tk/src/math.rs`) |

Operator sugar (`tk/src/ops.rs`) routes to the same calls, so a body reads as math: the
online-softmax update in `tk/src/kernels/fa.rs` is

```rust
let scale_vec = (max_vec_last - &max_vec).exp2();
o_reg = o_reg * &scale_vec;
norm_vec = norm_vec * &scale_vec;
let att = (att - &max_vec).exp2();
```

`T op &T` is same-shape, `RT op &RV` broadcasts the vector along the tile's layout axis,
`T op f64` is a scalar.

---

## Loops

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

Two rules the loop scope exists to make unforgettable:

- A per-iteration re-init (`g.zero(acc)` at the top of the body) must depend on the loop
  counter, or the linearizer hoists it above the loop and the accumulator carries stale
  state. Write `g.zero(lp.reinit(acc))`.
- A `RANGE` admits exactly one `END`. With several accumulators in one loop, chain the
  others into the one closing store (the GEMM threads each accumulator's A input through the
  previous accumulator's MMA), then read each final value as `acc.after(&ended)`.

`Kernel::range` / `range_uop` / `endrange` / `endrange_to` / `endrange_barrier_to` are the
raw forms `Loop` wraps; they emit the identical graph.

---

## Finishing and launching

```rust
pub fn finish(&self, stores: usize) -> Arc<UOp>          // tk/src/kernel.rs
```

`finish(n)` pops the last `n` terminal stores — one per output global — closes any still-open
tracked range around each, and sinks them with
`KernelInfo { opts_to_apply: Some(vec![]), name: Some(name) }`. A kernel that leaves a range
open at `finish` must have a single store. Stores reach the stack through the movement ops
or explicitly via `ker.push_store(store, buf)` (the straight-line norm kernel groups its
vector stores that way).

```rust
// tk/src/launch.rs
pub fn graph_launch(name, grid, block, out: Tensor, ins: &[&Tensor], caps: ArchCaps, build) -> Result<Tensor>
pub fn graph_launch_multi(name, grid, block, outs: Vec<Tensor>, ins, caps, build) -> Result<Vec<Tensor>>
pub fn launch_custom<T>(device, archs: ArchSet, validate, applies, build) -> Result<Option<T>>
pub fn run_kernel(name, grid, block, outs: &mut [&mut Tensor], ins: &[&Tensor], build) -> Result<()>
pub fn compile_kernel(name, grid, block, outs, ins, build) -> Result<CompiledLaunch>
```

`graph_launch` wraps the `SINK` as an `Op::Call` node and returns a lazy tensor; `out` is
`Tensor::empty(shape, dtype)`, and the placeholders the body sees are `[out, ins...]` — the
`bind_abi` order. `launch_custom` is the three-way policy every library kernel follows
([Authoring into the IR](./lowering)): `resolve_supported_arch` against the kernel's
`ArchSet` (`Ok(None)` off it), `validate(arch)` (`Err` for a malformed request),
`applies(arch)` (`Ok(None)` when the shape does not tile), then `build(arch)`.

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

`run_kernel` / `compile_kernel` are the direct-dispatch DEBUG face; see
[Debugging](./debugging).

---

## Below the tiles

Some kernels need addresses, not tiles. `tk/src/index.rs` is the flat-addressing layer every
tile op is built on, and it is public: `Idx` (`Const(i64)` or `Uop`), `flat_index(buf,
shape, idxs)`, `load_at`, `load_off`, `load_off_vec(buf, off, w)` / `store_off_vec` (one
`w`-wide access per lane that the renderer folds to a 128-bit instruction), and the gated
forms `load_off_gated` / `index_off_gated`. The RMS-norm kernel and the Qwen3
QKV-norm-RoPE prologue in `model/src/qwen3/tk/mod.rs` are written entirely at this level —
one wave per row, no `RANGE`, no LDS — and reuse the norm's row vocabulary (`plan`, `vload`,
`vpick`, `vstore`, `inv_rms`, `scale_by`).

`tk/src/asm.rs` exposes the AMD machine-scheduler controls as typed `Op::Custom` nodes
threaded on a dependency: `s_setprio(prio, dep)`, `s_waitcnt_lgkmcnt(n, dep)`,
`sched_barrier(mask, dep)`, `iglp_opt(mode, dep)`. The GEMM uses `sched_barrier(0, ..)` on
gfx12, where `ArchCaps::needs_pipeline_commit_fence()` says the backend scheduler would
otherwise hoist the pipeline's LDS commit above the trip's MMAs.

`tk/src/grid.rs::l2_swizzle(wgid, num_wgs, grid_m, grid_n)` maps a flattened workgroup id
to `(pid_m, pid_n)` so co-scheduled workgroups share an XCD's L2 (HipKittens' chiplet
transform); `GemmCfg::l2_swizzle` turns it on.
