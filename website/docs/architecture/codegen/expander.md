---
sidebar_label: Expander & reductions
---

# Expander and Reduction Lowering (stages 08–11)

The first four post-optimization stages take the optimizer's output — a kernel whose `RANGE`s now carry `Upcast`/`Unroll`/`Global`/`Local`/`GroupReduce` axis types — and make the intent concrete: unrolled ranges become shaped constants, `REDUCE` becomes an accumulator loop, local `STAGE`s become local buffers. All of them run inside `apply_post_optimization_configured_with_capture` (`optimizer/mod.rs`).

## 08 — post-opt symbolic

`POST_OPT_SYM = sym() + pm_move_where_on_load() + pm_flatten_range() + pm_reduce_unparented()`, one top-down fixpoint. Source order matters: later groups consume what earlier ones produce.

- `sym()` is the full tier-3 simplifier ([algebraic simplification](../optimizations/algebraic-simplification.md)).
- `pm_move_where_on_load` (`symbolic/patterns.rs`) rewrites `WHERE(cond, INDEX(buf, idx), 0)` into `INDEX(buf, WHERE(cond', idx, Invalid))`. The condition is split on `AND`; a clause moves into the index only if all its ranges are in scope of the `INDEX` and it has no `INDEX` dependency of its own; the remaining clauses stay in an outer `WHERE`. The inverted form `WHERE(cond, 0, INDEX(..))` is handled with the negated condition. Validity now rides inside the index expression, where the devectorizer and `indexing_simplify` can see it; it becomes a LOAD/STORE `gate` only at `19e`.
- `pm_flatten_range` rebuilds `END`/`REDUCE` range lists.
- `pm_reduce_unparented` drops reduce ranges the body does not reference: `Add` multiplies by the extent, `Mul` raises to the extent, `Max` just drops the range (there is no `Min` arm; `Min` reductions are not matched).

## 09 — expander (`pre_expand`)

`expander2() + pm_flatten_range() + mop_cleanup_patterns()` with a `RangeMap` context (`expand.rs`). `build_range_map` assigns every `Upcast`/`Unroll` `RANGE` a coordinate position in toposort order; the map's length is the rank of the shaped values this stage creates.

Three rules, in source order:

| Rule | Effect |
|------|--------|
| `Reduce { .. }` → `expand_reduce` | A loop-form `REDUCE` whose range list contains shaped non-`RANGE` entries turns those entries' axes (extent > 1) into leading *horizontal* axes: the source is permuted so they come first and `num_axes` counts them; the result is reshaped to keep size-1 placeholders. |
| `Range { axis_type: Upcast \| Unroll }` → `expand_range` | The range becomes `RESHAPE(STACK(CONST(0), ..., CONST(end-1)), shape)` where `shape` is all 1s except the range's own coordinate. Every consumer of the range becomes shaped by broadcasting; nothing is duplicated yet. |
| `Wmma { metadata.upcast_axes: Some(..) }` → `expand_wmma` | `contract_axis` moves the A/B upcast coordinates to the tail and flattens them into the fragment operands; `unroll_axis` restores the C coordinates on the output. The metadata's `upcast_axes` is cleared. |

`mop_cleanup_patterns` (`devectorize.rs`) is Tinygrad's `mop_cleanup`: merge nested `RESHAPE`s, drop identity `RESHAPE`/`PERMUTE`, merge `PERMUTE` chains, collapse `STACK(INDEX(b,0), INDEX(b,1), ..)` back to `b`, fold `INDEX(STACK(..), const)` to the lane, and compose `INDEX(INDEX(b, i), j)` into `INDEX(b, i, j)` when the indices are scalar. No symbolic matcher runs here.

For the worked example the reduce range `R2` (`Unroll`, extent 4) disappears and the index becomes shaped:

```text
[151] REDUCE(Add, num_axes=1, ranges=[118]) : Scalar(Float32) shape=[]
├── [149] INDEX : Scalar(Float32) shape=[Const(4)]
│   ├── [87] PARAM(slot=1) : Scalar(Float32) shape=[Const(512)]
│   └── [148] Add : Scalar(WeakInt) shape=[Const(4)]
│       ├── [147] Add : Scalar(WeakInt) shape=[Const(4)]
│       │   ├── [119] Mul : Scalar(WeakInt) shape=[]          ← R0 * 4
│       │   └── [146] STACK(len=4) : Scalar(WeakInt) shape=[Const(4)]
│       └── [90] Mul : Scalar(WeakInt) shape=[]              ← R1 * 64
└── [118] RANGE(R0, Reduce)
```

`expand_reduce` has already turned the 4-wide lane axis into `num_axes=1`, so the reduction over the lanes is horizontal and the remaining loop is over `R0` only.

:::tip[STACK is the only vector op]
A shaped value is a `STACK` of lanes (possibly nested, possibly behind a `RESHAPE`). `INDEX(STACK(..), c)` selects a lane with the same op that addresses a buffer. There is no vectorize/contract op pair, and `Upcast`/`Unroll` are `AxisType`s, not ops.
:::

## 10 — reduction lowering (`pm_reduce`)

`movement_cleanup_patterns() + pm_reduce_local()` with a `ReduceContext`. `movement_cleanup_patterns` is `mop_cleanup_patterns` plus two devectorizer-only rules (`RESHAPE(STACK([x]))` → `x` when shapes agree; a `RESHAPE` that only adds leading 1-dims → one `STACK([..])` wrapper per added dim).

`pm_reduce_local` (`devectorize.rs`) composes, in order:

1. **`pm_wmma_add`** — `WMMA(a, b, c) + add` → `WMMA(a, b, c + add)`, also through a `PERMUTE` and a `PERMUTE(RESHAPE(..))` wrapper that `expand_wmma` left on the output. `try_add` declines on a dtype mismatch instead of asserting.
2. **`pm_group_for_reduce`** (`expand.rs`) — a `REDUCE` with `GroupReduce` ranges becomes: partial `REDUCE` over the other ranges → `STAGE` of the partial with the in-scope `Local` ranges plus the group ranges (`BufferizeOpts::local_for_axis`) → `INDEX` of that stage with the locals and fresh `Reduce` loops (`axis_id.group_reduce_loop()`) → final `REDUCE` over those loops.
3. **`reduce_to_acc`** — a `REDUCE` with ranges. If `num_axes > 0` the lanes are first folded left-to-right in row-major order (`horizontal_reduce`). Then:

   ```text
   acc        = BUFFER(slot, AddrSpace::Reg)                       // placeholder_like(red)
   acc_init   = STORE(AFTER(acc, input_ranges), identity)           // 0 for Add, 1 for Mul, dtype min/max for Max/Min
   acc_loop   = AFTER(acc, [acc_init, reduce_ranges..])
   body       = op(acc_loop, horizontal_inp)                        // Add/Mul/Max; float Min is -(max(-a, -b))
   store_end  = END(STORE(acc, body), reduce_ranges)   tag=TAG_MERGEABLE
   result     = AFTER(acc, [store_end])
   ```

   `input_ranges` are the ranges in scope at the input that are neither reduced nor already ended, so the init lands inside the enclosing loops. There is no loop construct: the `END` closes the reduce ranges and the `AFTER` chain is the data dependency.
4. **`expand_horizontal_reduce`** — a `REDUCE` with no ranges left is the lane fold alone.
5. **END merging** — at the `SINK`, `merge_reduce_ends` groups the `TAG_MERGEABLE` `END`s by their reduce-range set and nesting context and replaces each group by `END(GROUP(computations), ranges)`; groups at a different nesting depth get cloned `RANGE`s with fresh axis ids so a range is closed by exactly one `END`.
6. **`clean_up_group_sink`** — single-source `GROUP`s unwrap; `NOOP`/`STACK`/`SINK`/`GROUP` sources of a `SINK` or `GROUP` are flattened.

`Min` is lowered through `Max` on floats (`-(max(-a, -b))`) so NaN behaves as in a max reduce; on integers it is `WHERE(a < b, a, b)`.

## 11 — local buffers

`pm_add_local_buffers = { Stage => add_local_buffer } + movement_op_patterns` (`optimizer/mod.rs`). Every `STAGE` that survived to this point is one `pm_group_for_reduce` just created (global ones became `STORE`s at the cut, and `bufferize_to_store` skipped `Local` ones on purpose). `add_local_buffer` allocates `UOp::placeholder(max_shape, dtype, slot, opts.addrspace)` — the slot is `LocalBufferContext::axis_slot` of the group axis, a deterministic hash for nested axis paths — and rewrites the stage into `AFTER(buffer, [END(STORE(INDEX(buffer, ranges), compute), ranges)])`. `movement_op_patterns` then pushes any movement op the new `INDEX` sits under into the index expression.

Tinygrad lowers reductions before adding local buffers for the same reason: the grouped-reduce stage does not exist until step 2 of `pm_reduce_local` has run.
