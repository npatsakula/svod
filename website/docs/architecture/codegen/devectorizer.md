---
sidebar_label: Devectorizer & index lowering
---

# GPU Dimensions, Devectorization and Index Lowering (stages 12–18)

After reduction lowering the kernel still speaks in shaped values and `WeakInt` indices. These stages map ranges to hardware indices, make every memory access an explicit scalar `LOAD`/`STORE`, re-widen contiguous accesses, and commit the index dtype. Source: `gpudims.rs`, `devectorize.rs`, `late/coalesce.rs`, `symbolic/index_lowering.rs`.

## 12 — GPU dimensions

Two matchers. `pm_lower_device_ranges` runs for every renderer: a `Device` range becomes the scalar `PARAM` variable `_device_num` with the range's bounds, and an `END` that closed it drops that entry. `pm_add_gpudims` runs only when `renderer.has_local || renderer.has_threads`; it matches the `SINK` once (`GpuDimsContext` remembers the lowered sink id so the engine's re-visit is a no-op).

`add_gpudims`:

1. Collects every `RANGE` keyed by `(axis_id, axis_type)`; bails if a `SPECIAL` already exists.
2. Global dims = `Global` and `Thread` axes; local dims = `Local`, `Warp`, `GroupReduce`. Both sorted by axis id; the `Warp` axis is moved to the front of the locals so it owns the low bits of the linear thread index (`mma.sync` addresses fragments by hardware lane).
3. Builds the index expressions:
   - `has_threads` (CPU): exactly one global axis and no locals, else the pass declines with a warning. The axis becomes `PARAM("core_id", 0..N-1)`.
   - `KernelInfo.dont_use_locals`: globals only, `get_grouped_dims("idx", ..)`.
   - otherwise `lidx*` from the local shape under `local_max_axes()` (or `local_max` per axis; the leading cap is pinned to the warp extent so nothing else folds into `lidx0`), then `gidx*` from the global shape under `global_max`, further capped by `global_prod_max / hardware_local_extents` when the renderer declares a work-item product limit.
4. `get_grouped_dims` is Tinygrad's: if the dims do not fit the per-axis caps, `group_dims` merges adjacent dims whose product fits; if nothing could be grouped, `split_dims` factors an oversized dim by its smallest divisor into the next slot. Either failure panics at scheduling time rather than at codegen (`"cannot limit dims to N axes"`). The result is one `SPECIAL(end, "gidxN")` per limited dim; when grouping or splitting happened, each original dim is reconstructed from the flat index with `FloorDiv`/`FloorMod` and simplified with `symbolic`. Global indices are produced with `reverse = true` (the recursion reverses the input *and* the output, so the names stay in iteration order).
5. **Store masking** (`compute_store_masks`): a `STORE` to global memory whose index is not in scope of every local range gets `WHERE((l1 == 0) & (l2 == 0) & .., idx, Invalid)` on its index, so only one work-item per unused local axis writes. The mask stays inside the index expression so the RANGE → SPECIAL substitution carries it to the hardware index.
6. Substitutes every GPU range with its index; `Reduce` ranges stay loops.

On the CPU example on the [worked example](./worked-example.md) page nothing happens: the row axis is `Weak` (no `Thread` axis was created) and the renderer has no locals.

## 13 — loads

`PM_ADD_LOADS = symbolic_simple() + pm_expand_broadcast() + pm_add_loads()`.

- `pm_expand_broadcast` starts with `pm_wmma_add` again, then makes broadcasting explicit: a `Binary`/`Ternary`/`STORE` whose sources have different shapes gets each source `RESHAPE`d (leading 1s) and `EXPAND`ed to the broadcast shape; a `WMMA` whose operand prefixes differ is expanded per output coordinate (`broadcast_and_devec_wmma`).
- `pm_add_loads` wraps every operand *consumed as a value* in `LOAD`: the sources of ALU ops, casts, `REDUCE`, `WMMA` and `STACK` that have an address space (`maybe_load`), and a `STORE` value that is itself an address. An `INDEX` used as an address — the `STORE` target, a `WMMA` fragment pointer — stays bare. The accumulator reads created at stage 10 (`AFTER(acc, ..)`) become `LOAD(AFTER(acc, ..))` here.

## 14 — devectorize

`devectorize()` is one `graph_rewrite` over `symbolic_simple + devectorize_patterns + bool_storage_patterns + indexing_simplify` (`Renderer` context, unused by the rules). There is no outer loop: the engine re-matches every replacement.

`devectorize_patterns` (`devectorizer2` in Tinygrad), in source order:

| Group | Rules |
|-------|-------|
| `movement_cleanup_patterns` | `mop_cleanup_patterns` plus the `RESHAPE(STACK([x]))` and leading-singleton materializations |
| `movement_op_patterns` | the rangeify movement rules (through `INDEX`, `AFTER`, `END`) |
| `no_vectorized_alu` | every unary/binary/ternary op, `CAST`, `BITCAST` with a non-empty shape → `devectorize_alu` |
| `mixed_representation_alu` | an ALU whose sources mix `STACK` and vector-dtype values: vector sources are unpacked into `STACK(INDEX(src, lane)..)`, then `devectorize_alu` |
| shaped `LOAD` / `STORE` | → `devectorize_alu` (per-lane `LOAD(INDEX)`; per-lane stores collected in a `GROUP`) |
| `INDEX(buf, [])` | → `buf` |
| `WMMA` | `stack_wmma_sources`: operands become `STACK`s of loaded lanes |
| `INDEX(buf, STACK(i0, i1, ..))` on a `PARAM`/`BUFFER` | → `STACK(INDEX(buf, i0), INDEX(buf, i1), ..)` — lanes stay addresses; the enclosing `LOAD`/`STORE` materializes them |
| `INDEX(buf, RESHAPE(i))` | → `RESHAPE(INDEX(buf, i))` |
| `RESHAPE` of a `Void` value | → the value (shape bookkeeping around `AFTER`/`STORE`) |
| one-element shaped value reshaped to scalar | → `INDEX(src, 0)` |
| `EXPAND` | `materialize_stack_broadcast` (a `STACK([x])` broadcast to N lanes → `STACK([x; N])`) or `expand_scalar_to_stack` |

`devectorize_alu` is Tinygrad's `do_devectorize`: it requires every source to have the result shape (or to be an `Invalid` base, whose scalar is polymorphic), enumerates the coordinates of the static shape, builds one scalar op per coordinate with `INDEX(source, c0, c1, ..)` operands, and reassembles with `stack_with_shape` (nested `STACK`s mirroring the shape) — or `GROUP` for a `STORE`. The lane count is the full product of the shape; there is no per-device fold width. Re-vectorization is the backend's job (LLVM's SLP vectorizer, or `memory_coalescing` two stages later for memory).

`bool_storage_patterns`: a bool `STORE` casts to `uint8`, a bool `LOAD` loads `uint8` and casts back, a `BITCAST` touching bool becomes a `CAST`. LLVM's `i1` may carry garbage in the upper bits.

`indexing_simplify` (`late/coalesce.rs`): for `INDEX(buf, WHERE(valid, idx, Invalid))`, `uop_given_valid` rewrites `idx` under the assumption `valid` holds (`symbolic/valid_simplification.rs`); the two-coordinate image form additionally drops validity clauses that the image bounds already imply (`drop_valid_stmts`).

After this stage every ALU op is scalar. In the worked example the four lanes of the index expression become four `LOAD(INDEX(PARAM, R0*4 + R1*64 + k))` and the horizontal `Add` chain is explicit.

## 15 — early symbolic

`sym()` once more, now on scalar code. Its reason for existing is the next stage: index expressions must be in canonical `base + const` form before coalescing can group them.

## 16 — memory coalescing

`memory_coalescing` (`late/coalesce.rs`) is a graph walk, not a matcher. It groups ungated `LOAD`s and `STORE`s by `(op, buffer, index base, validity)`, where the index is split into `base + integer_offset` (an `Invalid` or constant index is its own base). Within a group, consecutive offsets form runs; each run is cut into the widest fold length that divides the base offset:

- image buffers: 4;
- `supports_float4` renderers: powers of two down from `16 / sizeof(dtype)` when `access_bytes() >= 16` (eight 16-bit lanes, four `f32`), else from 4;
- otherwise scalar only. `Reg` buffers and non-foldable dtypes (anything but f32/f16/bf16/i32/u32/fp8) stay scalar.

A fold of width `n > 1` becomes `LOAD(SHRINK(buf, offset, n))` with the old loads replaced by `INDEX(load, lane)`, or `STORE(SHRINK(..), STACK(values))`. `SHRINK` carries the group shape; the memory dtype stays scalar. `DMC=1` disables the pass. In the worked example the four unrolled loads become one `LOAD(SHRINK(PARAM(1), R0*4 + R1*64, 4))`, which the LLVM backend renders as `load <4 x float>`.

## 17 — bottom-up elementwise / image pass

`symbolic_simple + no_vectorized_alu + pm_simplify_add_image`, applied with `graph_rewrite_bottom_up` and an `AddImageContext`. The image rules canonicalize f16 accesses to f32 image buffers (`LOAD` → `LOAD.cast(f16)`, `STORE(value.cast(f32))`, drop a `CAST(CAST(x, f16), f32)` round-trip). Image buffer *creation* has no Svod target; the rules only serve existing image accesses. `no_vectorized_alu` runs again because an image rewrite can reintroduce a shaped op.

## 16 — extra symbolic

`extra_symbolic_patterns = sym() + indexing_simplify()`. Indices are still `WeakInt` here on purpose: the distributive and index-validity rules of `sym` and `indexing_simplify` need the weak dtype, so this is their last chance. (The label collides with memory coalescing's; both print under `SVOD_DUMP_STAGE=16`.)

## 17 — index dtype lowering

`lower_index_patterns = symbolic_simple + pm_fold_cast_const + pm_lower_index_dtype + indexing_simplify`, with one `WeakMemo` per kernel (Tinygrad's single `ctx={}`). Port of `tinygrad/uop/weak.py`.

`select_dtype(u)`: `WeakFloat` → the default float; an integer whose `vmin`/`vmax` fit `i32` → the default int, otherwise `Int64`; vector count preserved.

`pm_lower_index_dtype` composes:

1. `pm_commit_weak` — a `Binary`/`Ternary` with a weak source and a non-weak `least_upper_dtype` commits the weak sources to it (`commit_weak`: a `CONST` is retyped, anything else is cast); a `STORE` whose value is weak commits it to the index's dtype.
2. `pm_cast_weak` — `CAST(weak_alu, concrete)` pushes the concrete dtype into the ALU's sources.
3. `SHRINK` offsets/sizes are committed with `select_dtype`.
4. Any non-weak node with weak sources → `lower_weak_srcs`: each weak source is rewritten with `pm_lower_weak` (memoized by source id) and the trailing weak `CAST` is absorbed by the consumer's own edge. `pm_lower_weak` is the three-phase cascade:
   - leaves: `CONST`/`VCONST`/scalar `PARAM` become `concrete.cast(weak)`;
   - `Unary`, `Binary`, `WHERE` (condition skipped), `RANGE`, `STACK`, `SPECIAL` (`lower_weak_node`): unwrap the weak casts on the sources, compute the concrete dtype (`least_upper_dtype` of `select_dtype(u)` and the sources for binary ops; `dtype_from_op` otherwise), cast every source to it, keep a weak `CAST` on the result unless it is a `STACK`;
   - `INDEX` with a weak dtype: the buffer is cast to the selected dtype and each weak index committed;
   - `CAST(weak, CAST(weak, x))`: the inner cast is committed, the outer kept.
5. An `INDEX` (or `SHRINK`) whose gated index already came out as `Int64` is narrowed back to `Int32` when the buffer's element count fits `i32`.

This is the stage where `WHERE(valid, idx, Invalid)` keeps its shape: `Invalid` sources are left alone by `lower_weak_node`. In the worked example every `WeakInt` becomes `Int32` (`RANGE(R0, Reduce) : Scalar(Int32)`), and the `PARAM` sizes become `Int32` constants.

## 18 — final symbolic

`symbolic()` (tier 2, no `pm_simplify_valid`/lane folds) on the concretely typed graph. With `SVOD_SPEC` on, `verify_no_legacy_index_dtype` asserts no `WeakInt` survived.
